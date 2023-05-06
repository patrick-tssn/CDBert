import os
import collections
from pathlib import Path
from packaging import version
import numpy as np
from tqdm import tqdm
import logging
import shutil
from pprint import pprint
import random
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

from transformers import BertConfig, BertModel

from utils.param import parse_args
from utils.dist_utils import all_gather
from utils.utils import load_state_dict, LossMeter, set_global_logging_level, init_logger, logger
from models.trainer_base import TrainerBase
from cclue.cclue_key_eg_data import get_loader
from cclue.cclue_key_eg_model import BertDictMRC

# set_global_logging_level(logging.ERROR, ["transformers"])
proj_dir = Path(__file__).resolve().parent.parent

def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True, num_labels=0):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        model_kwargs = {}
        model_class = BertDictMRC
        self.num_labels = num_labels
        config = self.create_config()
        self.model = BertDictMRC(config)
        self.tokenizer = self.create_tokenizer()
        self.model.bert = self.create_model(BertModel, config, **model_kwargs)
        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        if not self.args.debug:
            self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
    
    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
                
            
            # if 'bert-base-chinese' not in ckpt_path:
            # # if key.startswith('dict_bert'):
            #     new_key = 'dict_bert' + key
            #     state_dict[new_key] = state_dict.pop(key)

        results = self.model.dict_bert.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)
    
    def create_config(self):
        config_class = BertConfig

        config = config_class.from_pretrained(self.args.backbone, num_labels=self.num_labels)
        config.radical_vocab_size = self.args.rid+1
        config.fuse = self.args.fuse
        config.glyph = self.args.glyph


        return config


    
    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model

    def train(self):
        seed_everything()
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_test = 0.
            best_epoch = 0


            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)


        if self.args.distributed:
            dist.barrier()
            
        if self.args.evaluate_start:
            score_dict = self.evaluate(self.test_loader)

            test_score = score_dict['accuracy'] * 100.
            test_score_raw = score_dict['accuracy']

            log_str = ''
            log_str += "\nZero shot: Test Raw %0.2f Topk %0.2f" % (test_score_raw, test_score)

            print(log_str)
            logger.info(log_str)
            exit(0)

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=80)

            epoch_results = {
                'loss': 0.,

            }

            # quesid2ans = {}
            textid2ans = {}

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                    desc_str += f' | Loss {loss_meter.val:4f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()

            # Validation
            score_dict = self.evaluate(self.val_loader)

            if self.verbose:
                valid_score = score_dict['accuracy'] * 100.
                valid_score_raw = score_dict['accuracy']
                # valid_score = score_dict['topk_score'] * 100.
                # valid_score_raw = score_dict['overall']
                if valid_score_raw > best_valid or epoch == 0:
                    best_valid = valid_score_raw
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''
                log_str += "\nEpoch %d: Valid Raw %0.2f Topk %0.2f" % (epoch, valid_score_raw, valid_score)
                log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)

                wandb_log_dict['Valid/score'] = valid_score

                print(log_str)
                logger.info(log_str)
                
            # Test
            score_dict = self.evaluate(self.test_loader)

            if self.verbose:
                test_score = score_dict['accuracy'] * 100.
                test_score_raw = score_dict['accuracy']
                # valid_score = score_dict['topk_score'] * 100.
                # valid_score_raw = score_dict['overall']
                if test_score_raw > best_test or epoch == 0:
                    best_test = test_score_raw
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''
                log_str += "\nEpoch %d: Test Raw %0.2f Topk %0.2f" % (epoch, test_score_raw, test_score)
                log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_epoch, best_test)

                print(log_str)
                logger.info(log_str)
                
            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")
            
        # # save Dev Set
        # best_path = os.path.join(self.args.output, 'BEST')
        # self.load(best_path)
        # quesid2ans = self.predict(self.val_loader)
        # evaluator = self.val_loader.evaluator
        # evaluator.dump_result(quesid2ans, os.path.join(self.args.output, 'tnews_dev_predict.json'))

        # Test Set
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)

        quesid2ans = self.predict(self.test_loader)

        if self.verbose:
            evaluator = self.test_loader.evaluator
            score_dict = evaluator.evaluate(quesid2ans)
            print(f'The ACC. of the best ckpt. for predict result is: {score_dict}')
            evaluator.dump_result(quesid2ans, os.path.join(self.args.output, 'predict.json'))


        if self.args.submit:
            dump_path = os.path.join(self.args.output, 'submit.json')
            self.predict(self.submit_test_loader, dump_path)



        if self.args.distributed:
            dist.barrier()
            exit()

    def predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            # quesid2ans = {}
            textid2ans = {}
            # res = []
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=80, desc="Prediction")
            for i, batch in enumerate(loader):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans']
                text_id = batch['text_id']
                # ques_ids = batch['question_ids']
                # label = batch['label']
                for tid, ans in zip(text_id, pred_ans):
                    textid2ans[tid] = ans
                # for qid, ans in zip(ques_ids, pred_ans):
                #     quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

        if self.args.distributed:
            dist.barrier()

        tid2ans_list = all_gather(textid2ans)
        if self.verbose:
            textid2ans = {}
            for tid2ans in tid2ans_list:
                for k, v in tid2ans.items():
                    textid2ans[k] = v
            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(textid2ans, dump_path)


        return textid2ans

    def evaluate(self, loader, dump_path=None):
        textid2ans = self.predict(loader, dump_path)

        if self.verbose:
            evaluator = loader.evaluator
            # acc_dict = evaluator.evaluate_raw(textid2ans)
            topk_score = evaluator.evaluate(textid2ans)
            # acc_dict['topk_score'] = topk_score
            acc_dict = {'accuracy':topk_score}

            return acc_dict

def main_worker(gpu, args):
    seed_everything()
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
    else:
        torch.cuda.set_device(args.gpu)


    print(f'Building train loader at GPU {gpu}')
    train_loader, num_labels = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader, num_labels = get_loader(
        args,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    print(f'Building test loader at GPU {gpu}')
    test_loader, num_labels = get_loader(
        args,
        split=args.test, mode='test', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True, num_labels=num_labels)

    if args.submit:
        print(f'Building test submit loader at GPU {gpu}')
        submit_test_loader = get_loader(
            args,
            split='test', mode='val', batch_size=valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
        )
        trainer.submit_test_loader = submit_test_loader

    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    init_logger(log_file=args.output + '/{}-{}.log'.format(args.model_name, args.task_name))
    
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        elif args.load_lxmert_qa is not None:
            ckpt_str = "_".join(args.load_lxmert_qa.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
    if args.debug:
        main_worker(0, args)
    else:
        main_worker(0, args)

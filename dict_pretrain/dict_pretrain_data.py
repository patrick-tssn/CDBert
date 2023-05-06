from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer, BertTokenizerFast

from utils.preprocess import corrupt_dict, corrupt_radical_dict


project_dir = Path(__file__).resolve().parent.parent 
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')



def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)


class PretrainCLDataset(Dataset):
    def __init__(self, data_path, rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args

        # Loading datasets to data
        if self.verbose:
            print('Data sources: ', data_path)

        
        with open(data_path) as jh:
            self.data = json.load(jh)
        self.n_data = len(self.data)
        
        if self.verbose:
            print("# examples:", len(self.data))

        if 'bert' in self.args.backbone:
            self.tokenizer = BertTokenizer.from_pretrained(args.backbone)

        self.glyph = self.args.glyph
        if self.glyph == 'radical':
            with open(self.args.radical_path) as jh:
                self.radical2idx = json.load(jh)
            self.rid = args.rid

        # self.rid = args.rid

    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid
        
        loss_weight = 1

        # preprocess original word
        word = datum['entry']
        sents = datum['descs']
        sent = sents[datum['desc']]
        eg = datum['eg']
        dict_eg = datum['dict_eg']
        if dict_eg:
            sent_eg = sent + '【例子】' + dict_eg
            # sent_egs = [x+'【例子】'+dict_eg for x in sents]
            sent_egs = sents
        else:
            sent_eg = sent
            sent_egs = sents
        mean_idx = datum['desc']
        syno_loss_weight = datum['syno_weight']
        anto_loss_weight = datum['syno_weight']
        eg_loss_weight = datum['eg_weight']
        
        # input
        if self.glyph == 'radical':
            input_ids, target_ids, input_radical_ids, target_radical_ids, token_types = corrupt_radical_dict(
                word, sent_eg, self.radical2idx, self.rid, self.tokenizer, self.args.max_text_length
            )

            assert len(input_ids) == len(input_radical_ids)
        else:
            input_ids, target_ids, token_types = corrupt_dict(
                word, sent_eg, self.tokenizer, self.args.max_text_length
            )
        # synonym
        syno_word = datum['syno_entry']
        syno_sent = datum['syno_desc']
        if self.glyph == 'radical':
            syno_input_ids, syno_target_ids, syno_input_radical_ids, syno_target_radical_ids, syno_token_types = corrupt_radical_dict(
                syno_word, syno_sent, self.radical2idx, self.rid, self.tokenizer, self.args.max_text_length
            )

            assert len(syno_input_ids) == len(syno_input_radical_ids)
        else:
            syno_input_ids, syno_target_ids, syno_token_types = corrupt_dict(
                syno_word, syno_sent, self.tokenizer, self.args.max_text_length
            )
        
        # antonym
        anto_word = datum['anto_entry']
        anto_sent = datum['anto_desc']
        if self.glyph == 'radical':
            anto_input_ids, anto_target_ids, anto_input_radical_ids, anto_target_radical_ids, anto_token_types = corrupt_radical_dict(
                anto_word, anto_sent, self.radical2idx, self.rid, self.tokenizer, self.args.max_text_length
            )

            assert len(anto_input_ids) == len(anto_input_radical_ids)
        else:
            anto_input_ids, anto_target_ids, anto_token_types = corrupt_dict(
                anto_word, anto_sent, self.tokenizer, self.args.max_text_length
            )
        
        # means
        eg_means_input = []
        eg_means_token_types = []
        eg_means_radical_input = []
        eg_means_mask = []
        eg_means_label = mean_idx
        # for s in sents:
        for s in sent_egs:
            if self.glyph == 'radical':
                eg_input_ids, eg_target_ids, eg_input_radical_ids, eg_target_radical_ids, eg_token_types = corrupt_radical_dict(
                    word, s, self.radical2idx, self.rid, self.tokenizer, self.args.max_text_length
                )
                
            else:
                eg_input_ids, eg_target_ids, eg_token_types = corrupt_dict(
                    word, s, self.tokenizer, self.args.max_text_length
                )
            eg_mask_ids = [1] * len(eg_target_ids)
            while len(eg_target_ids) < self.args.max_text_length:
                eg_target_ids.append(self.tokenizer.pad_token_id)
                eg_mask_ids.append(0)
                eg_token_types.append(0)
                if self.glyph == 'radical':
                    eg_target_radical_ids.append(self.tokenizer.pad_token_id)
            eg_means_input.append(eg_target_ids)
            eg_means_mask.append(eg_mask_ids)
            eg_means_token_types.append(eg_token_types)
            if self.glyph == 'radical':
                eg_means_radical_input.append(eg_target_radical_ids)
        example_input_ids = self.tokenizer.encode(eg, max_length=self.args.max_text_length)
            
        

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        
        # out_dict['source_text'] = source_text
        # out_dict['target_text'] = target_text
        out_dict['sent'] = sent
        out_dict['loss_weight'] = loss_weight


        # synonym
        out_dict['syno_input_ids'] = torch.LongTensor(syno_input_ids)
        out_dict['syno_input_length'] = len(syno_input_ids)
        out_dict['syno_target_ids'] = torch.LongTensor(syno_target_ids)
        out_dict['syno_target_length'] = len(syno_target_ids)
        out_dict['syno_loss_weight'] = syno_loss_weight
        


        # antonym
        out_dict['anto_input_ids'] = torch.LongTensor(anto_input_ids)
        out_dict['anto_input_length'] = len(anto_input_ids)
        out_dict['anto_target_ids'] = torch.LongTensor(anto_target_ids)
        out_dict['anto_target_length'] = len(anto_target_ids)
        out_dict['anto_loss_weight'] = anto_loss_weight
        
 
        
        # example
        out_dict['example_input_ids'] = torch.LongTensor(example_input_ids)
        out_dict['example_input_length'] = len(example_input_ids)
        out_dict['mean_input_ids'] = torch.LongTensor(eg_means_input)
        out_dict['mean_token_types'] = torch.LongTensor(eg_means_token_types)
        out_dict['mean_mask_ids'] = torch.LongTensor(eg_means_mask)
        out_dict['mean_label_ids'] = torch.LongTensor([int(eg_means_label)])
        out_dict['mean_input_length'] = len(eg_means_input)
        out_dict['eg_loss_weight'] = eg_loss_weight
        
        if self.glyph == 'radical':
            out_dict['radical_ids'] = torch.LongTensor(input_radical_ids)
            out_dict['target_radical_ids'] = torch.LongTensor(target_radical_ids)
            out_dict['syno_radical_ids'] = torch.LongTensor(syno_target_radical_ids)
            out_dict['anto_radical_ids'] = torch.LongTensor(anto_target_radical_ids)
            out_dict['means_radical_ids'] = torch.LongTensor(eg_means_radical_input)
            

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        # V_L = len(batch[0]['boxes'])

        # original 
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)


        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        entry_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        mask_ids = torch.zeros(B, T_W_L, dtype=torch.long)



        # synonym 
        syno_S_W_L = max(entry['syno_input_length'] for entry in batch)
        syno_T_W_L = max(entry['syno_target_length'] for entry in batch)
        # syno_G_W_L = max(entry['syno_glyph_length'] for entry in batch)
        syno_input_ids = torch.ones(B, syno_S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        syno_target_ids = torch.ones(B, syno_T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        syno_mask_ids = torch.zeros(B, syno_T_W_L, dtype=torch.long)

        # antonym 
        anto_S_W_L = max(entry['anto_input_length'] for entry in batch)
        anto_T_W_L = max(entry['anto_target_length'] for entry in batch)
        # anto_G_W_L = max(entry['anto_glyph_length'] for entry in batch)
        anto_input_ids = torch.ones(B, anto_S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        anto_target_ids = torch.ones(B, anto_T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        anto_mask_ids = torch.zeros(B, anto_T_W_L, dtype=torch.long)

        # example
        example_W_L = max(entry['example_input_length'] for entry in batch)
        example_input_ids = torch.ones(B, example_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        example_mask_ids = torch.zeros(B, example_W_L, dtype=torch.long)
        mean_W_L = max(entry['mean_input_length'] for entry in batch)
        mean_input_ids = torch.ones((B, mean_W_L, self.args.max_text_length), dtype=torch.long) * self.tokenizer.pad_token_id
        mean_mask_ids = torch.zeros((B, mean_W_L, self.args.max_text_length), dtype=torch.long)
        mean_label_mask_ids = torch.zeros((B, mean_W_L), dtype=torch.long)
        mean_label_ids = torch.ones((B, 1), dtype=torch.long)
        
        if self.glyph == 'radical':
            radical_ids = torch.zeros(B, T_W_L, dtype=torch.long)
            target_radical_ids = torch.zeros(B, T_W_L, dtype=torch.long)
            syno_radical_ids = torch.zeros(B, syno_T_W_L, dtype=torch.long)
            anto_radical_ids = torch.zeros(B, anto_T_W_L, dtype=torch.long)
            mean_radical_ids = torch.zeros((B, mean_W_L, self.args.max_text_length), dtype=torch.long)



        loss_weights = torch.ones(B, dtype=torch.float)
        syno_loss_weight = torch.ones(B, dtype=torch.float)
        anto_loss_weight = torch.ones(B, dtype=torch.float)
        eg_loss_weight = torch.ones(B, dtype=torch.float)
        
        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            entry_ids[i, :entry['target_length']] = entry['target_ids']
            

            # synonym
            syno_input_ids[i, :entry['syno_input_length']] = entry['syno_input_ids']
            syno_target_ids[i, :entry['syno_target_length']] = entry['syno_target_ids']
            syno_loss_weight[i] = entry['syno_loss_weight']

            anto_input_ids[i, :entry['anto_input_length']] = entry['anto_input_ids']
            anto_target_ids[i, :entry['anto_target_length']] = entry['anto_target_ids']
            anto_loss_weight[i] = entry['anto_loss_weight']

            
            # example
            example_input_ids[i, :entry['example_input_length']] = entry['example_input_ids']
            mean_input_ids[i, :entry['mean_input_length']] = entry['mean_input_ids']
            mean_mask_ids[i, :entry['mean_input_length']] = entry['mean_mask_ids']
            mean_label_mask_ids[i, :entry['mean_input_length']] = torch.ones(entry['mean_input_length'], dtype=torch.long)
            mean_label_ids[i, 0] = entry['mean_label_ids']
            eg_loss_weight[i] = entry['eg_loss_weight']
            
            if self.glyph == 'radical':
                radical_ids[i, :entry['input_length']] = entry['radical_ids']
                target_radical_ids[i, :entry['input_length']] = entry['target_radical_ids']
                syno_radical_ids[i, :entry['syno_target_length']] = entry['syno_radical_ids']
                anto_radical_ids[i, :entry['anto_target_length']] = entry['anto_radical_ids']
                mean_radical_ids[i, :entry['mean_input_length']] = entry['means_radical_ids']
                

            if 'ans' in entry:
                ans.append(entry['ans'])
            if 'task' in entry:
                tasks.append(entry['task'])
            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone or 'bert' in args.backbone
        # original
        word_mask = target_ids != self.tokenizer.pad_token_id
        mask_ids[word_mask] = 1
        dict_word_mask = input_ids == self.tokenizer.mask_token_id
        target_ids[~dict_word_mask] = -100
        # mask_ids[dict_word_mask] = 1
        # mask_ids[~dict_word_mask] = 0
        # synonym
        syno_word_mask = syno_target_ids != self.tokenizer.pad_token_id
        # syno_target_ids[~syno_word_mask] = -100
        syno_mask_ids[syno_word_mask] = 1
        # dict_word_mask = syno_input_ids == self.tokenizer.mask_token_id
        # syno_mask_ids[dict_word_mask] = 1
        # syno_mask_ids[~dict_word_mask] = 0
        # antonym
        anto_word_mask = anto_target_ids != self.tokenizer.pad_token_id
        # anto_target_ids[~anto_word_mask] = -100
        anto_mask_ids[anto_word_mask] = 1
        # dict_word_mask = anto_input_ids == self.tokenizer.mask_token_id
        # anto_mask_ids[dict_word_mask] = 1
        # anto_mask_ids[~dict_word_mask] = 0
        # example
        example_word_mask = example_input_ids != self.tokenizer.pad_token_id
        example_mask_ids[example_word_mask] = 1
        

        batch_entry['task'] = tasks
        # if self.args.glyph:
        #     target_ids = torch.cat((target_ids, torch.ones(glyph_mask_ids.size(), dtype=torch.long)*(-100)), dim=1)

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['entry_ids'] = entry_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['mask_ids'] = mask_ids
        # batch_entry['glyph_embeds'] = glyph_embeds
        # batch_entry['glyph_mask_ids'] = glyph_mask_ids

        # synonym
        batch_entry['syno_input_ids'] = syno_input_ids
        batch_entry['syno_target_ids'] = syno_target_ids
        batch_entry['syno_mask_ids'] = syno_mask_ids
        batch_entry['syno_loss_weight'] = syno_loss_weight
        
        # batch_entry['syno_glyph_embeds'] = syno_glyph_embeds
        # batch_entry['syno_glyph_mask_ids'] = syno_glyph_mask_ids
        # antonym
        batch_entry['anto_input_ids'] = anto_input_ids
        batch_entry['anto_target_ids'] = anto_target_ids
        batch_entry['anto_mask_ids'] = anto_mask_ids
        batch_entry['anto_loss_weight'] = anto_loss_weight
        # batch_entry['anto_glyph_embeds'] = anto_glyph_embeds
        # batch_entry['anto_glyph_mask_ids'] = anto_glyph_mask_ids
        
        # exampel
        batch_entry['example_input_ids'] = example_input_ids
        batch_entry['example_mask_ids'] = example_mask_ids
        batch_entry['mean_input_ids'] = mean_input_ids
        batch_entry['mean_mask_ids'] = mean_mask_ids
        batch_entry['mean_label_mask_ids'] = mean_label_mask_ids
        batch_entry['mean_label_ids'] = mean_label_ids
        batch_entry['eg_loss_weight'] = eg_loss_weight

        # batch_entry['boxes'] = boxes
        # batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences
        
        if self.glyph == 'radical':
            batch_entry['radical_ids'] = radical_ids
            batch_entry['target_radical_ids'] = target_radical_ids
            batch_entry['syno_radical_ids'] = syno_radical_ids
            batch_entry['anto_radical_ids'] = anto_radical_ids
            batch_entry['mean_radical_ids'] = mean_radical_ids
            

        return batch_entry




class PretrainDataset(Dataset):
    def __init__(self, data_path, rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args


        # Loading datasets to data
        if self.verbose:
            print('load data from: ', data_path)
        
        with open(data_path) as jh:
            self.data = json.load(jh)
        self.n_data = len(self.data)


        if self.verbose:
            print("# examples:", len(self.data))


        # self.n_boxes = args.n_boxes
        if 'bert' in self.args.backbone:
            self.tokenizer = BertTokenizer.from_pretrained(args.backbone)
        
        self.glyph = self.args.glyph
        if self.glyph == 'radical':
            with open(self.args.radical_path) as jh:
                self.radical2idx = json.load(jh)
            self.rid = args.rid

        


    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        loss_weight = 1
        word = datum['entry']
        sent = datum['desc']
        
        if self.glyph == 'radical':
            input_ids, target_ids, radical, _ = corrupt_radical_dict(
                word, sent, self.radical2idx, self.rid, self.tokenizer, self.args.max_text_length
            )

            radical_ids = radical
            assert len(input_ids) == len(radical_ids)
        else:
            input_ids, target_ids = corrupt_dict(
                word, sent, self.tokenizer, self.args.max_text_length
            )
     
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        # out_dict['source_text'] = source_text
        # out_dict['target_text'] = target_text
        # out_dict['task'] = task
        out_dict['sent'] = sent
        out_dict['loss_weight'] = loss_weight
        
        if self.glyph == 'radical':
            out_dict['radical_ids'] = torch.LongTensor(radical_ids)

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        # V_L = len(batch[0]['boxes'])

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)
        # G_W_L = max(entry['glyph_length'] for entry in batch)

        # feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        mask_ids = torch.zeros(B, T_W_L, dtype=torch.long)
        
        if self.glyph == 'radical':
            radical_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id


        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            
            if self.glyph == 'radical':
                radical_ids[i, :entry['input_length']] = entry['radical_ids']


            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone or 'bert' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        # target_ids[~word_mask] = -100
        batch_entry['task'] = tasks
        mask_ids[word_mask] = 1 # 1 not mask 0 otherwise

        dict_word_mask = input_ids == self.tokenizer.mask_token_id
        target_ids[~dict_word_mask] = -100
        
        # mask_ids[~dict_word_mask] = 0

        # if self.args.glyph:
        #     target_ids = torch.cat((target_ids, torch.ones(glyph_mask_ids.size(), dtype=torch.long)*(-100)), dim=1)

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['mask_ids'] = mask_ids
        
        if self.glyph == 'radical':
            batch_entry['radical_ids'] = radical_ids

        # batch_entry['glyph_embeds'] = glyph_embeds
        # batch_entry['glyph_mask_ids'] = glyph_mask_ids


        # batch_entry['boxes'] = boxes
        # batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        return batch_entry


def get_loader(args, data_path='dict', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1, cl=False):


    verbose = (gpu == 0)
    if cl:
        dataset = PretrainCLDataset(
            data_path,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            is_train=(mode == 'train'),
            )    
    else:
        dataset = PretrainDataset(
            data_path,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            is_train=(mode == 'train'),
            )
    # dataset = PretrainCLDataset(
    #     split,
    #     rank=gpu,
    #     topk=topk,
    #     verbose=verbose,
    #     args=args,
    #     is_train=(mode == 'train'),
    #     )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader


class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError

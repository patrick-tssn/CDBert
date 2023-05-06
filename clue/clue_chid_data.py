
from pathlib import Path
from collections import defaultdict
import json
import time
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import re
import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer


project_dir = Path(__file__).resolve().parent.parent  
workspace_dir = project_dir
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
CLUE_dir = dataset_dir.joinpath('CLUE/')



class CHIDDataset(Dataset):

    def __init__(self, split='train', verbose=True, args=None, max_length: int = 512, mode='train'):
        super().__init__()
        
        
        self.max_length = max_length
        self.max_text_length = args.max_text_length
        self.args = args
        with open(dataset_dir.joinpath('CLUE/{}/{}.json'.format(args.task_name, split)), 'r', encoding='utf8') as jh:
            lines = jh.readlines()
        print(args.task_name)
        # self.lines = lines
        # add text id
        self.lines = []
        self.id2datum = {}
        for idx, line in enumerate(lines):
            line = json.loads(line)

            if args.task_name == 'chid': # option context
                answers = line['choices']
                context_left = line['context_left']
                context_right = line['context_right']
                if 'test' not in split:
                    label = line['label']
                else:
                    label = '0'
                context_entrys = line['entrys']
                item_idx = line['item_idx']
                example_idx = line['example_id']
                self.lines.append({'text_idx':item_idx, 'example_id':example_idx, 'context_left':context_left, 'context_right':context_right, 'answers':answers, 'label':str(label), 'entrys':context_entrys})
                self.id2datum[item_idx] = {'label':str(label)}
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.backbone,
        )
        labels = self.get_labels()
        self.labels2id = {value: key for key, value in enumerate(labels)}
    
        if 'large' in self.args.backbone:
            self.hd_sz = 1024
        else:
            self.hd_sz = 768
        

        start = time.time()

        
        self.entry_dict = torch.load(dataset_dir.joinpath(args.embedding_lookup_table + 'entry_lookup_embedding.pt'))

        print('load embedding data cost {}'.format(time.time()-start))
        

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        # text_idx,label,sentence,entrys = line['text_idx'],line['label'],line['sentence'],line['entrys']
        text_idx,example_idx,label,context_left,context_right,answers,entrys = line['text_idx'],line['example_id'],line['label'],line['context_left'],line['context_right'],line['answers'],line['entrys']
        # entrys
        label = self.labels2id[label]
        # sentence = sentence[:self.max_seq-2]
        # convert characters to input ids
        context_left = re.sub('#idiom\d{6}#','[MASK][MASK][MASK][MASK]',context_left)
        context_right = re.sub('#idiom\d{6}#','[MASK][MASK][MASK][MASK]',context_right)
        context_left_tokens = self.tokenizer.tokenize(context_left)
        context_right_tokens = self.tokenizer.tokenize(context_right)
        
        # round context tokens
        max_tokens = self.args.max_text_length - 3 - 5
        num_l = max_tokens // 2
        num_r = max_tokens - num_l
        num_tokens = len(context_left_tokens) + len(context_right_tokens)
        pos = len(context_left_tokens)

        if pos >= num_l and (num_tokens - 1 - pos) >= num_r:
            context_left_tokens = context_left_tokens[-num_l:]
            context_right_tokens = context_right_tokens[:num_r]
        elif pos <= num_l:
            right_len = max_tokens - len(context_left_tokens)
            context_right_tokens = context_right_tokens[:right_len]
        elif (num_tokens - 1 - pos) <= num_r:
            left_len = max_tokens - len(context_right_tokens)
            context_left_tokens = context_left_tokens[-left_len:]
        
        input_token_ids = []
        input_token_types = []
        input_attention_masks = []
        input_entry_hidden_states = []
        for answer in answers:
            # if answer in self.dict_set:
            #     entrys = [answer] + entrys
            entrys = [answer] + entrys
            answer_tokens = self.tokenizer.tokenize(answer)
            bert_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + answer_tokens + ['[SEP]'] + context_left_tokens + ['[unused1]'] + context_right_tokens + ['[SEP]'])
            token_type_ids = [0] * len(bert_tokens)
            attention_mask_ids = [1] * len(bert_tokens)
            # pad to max text length
            while len(bert_tokens) < self.max_text_length:
                bert_tokens.append(self.tokenizer.pad_token_id)
                token_type_ids.append(self.tokenizer.pad_token_id)
                attention_mask_ids.append(self.tokenizer.pad_token_id)
            input_token_ids.append(bert_tokens)
            input_token_types.append(token_type_ids)
            input_attention_masks.append(attention_mask_ids)
                

        # convert token_id to embeds
        glyph_ids = []
        desc_ids = []
        out_dict = {}
        
        for ii, entry in enumerate(entrys):
            if entry in self.entry_dict.keys():
                desc_ids.append(self.entry_dict[entry])
            
        if desc_ids != []:
            out_dict['entry_ids'] = torch.cat(desc_ids)[:self.args.max_text_length]
            out_dict['entry_length'] = len(out_dict['entry_ids'])
            
        else:
            out_dict['entry_ids'] = torch.zeros(1, self.hd_sz)
            out_dict['entry_length'] = 0


        input_ids = torch.LongTensor(input_token_ids)
        input_types = torch.LongTensor(input_token_types)
        input_masks = torch.LongTensor(input_attention_masks)
        label = torch.LongTensor([int(label)])

        out_dict['input_ids'] = input_ids
        out_dict['input_types'] = input_types
        out_dict['input_masks'] = input_masks
        out_dict['input_length'] = len(input_ids)
        out_dict['label'] = label
        out_dict['text_id'] = text_idx
        out_dict['example_id'] = example_idx

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        # args = batch[0]['args']
        B = len(batch)
        
        input_ids = torch.ones((B, self.args.choices, self.args.max_text_length), dtype=torch.long) * self.tokenizer.pad_token_id
        input_types = torch.ones((B, self.args.choices, self.args.max_text_length), dtype=torch.long) * self.tokenizer.pad_token_id
        input_mask = torch.zeros((B, self.args.choices, self.args.max_text_length), dtype=torch.long)
        label_ids = torch.ones(B, 1, dtype=torch.long)
        
        E_W_L = max(entry['entry_length'] for entry in batch)
        entry_ids = torch.zeros((B, E_W_L, self.hd_sz), dtype=torch.float)
        entry_mask = torch.zeros(B, E_W_L, dtype=torch.long)

        text_ids = []
        example_ids = []
        desc_batch_ids = []
        for i, entry in enumerate(batch):
            input_ids[i] = entry['input_ids']
            input_types[i] = entry['input_types']
            input_mask[i] = entry['input_masks']
            label_ids[i, 0] = entry['label']
            # clip_embeds[i, :entry['input_length']] = entry['clip_embeds']
            text_ids.append(entry['text_id'])
            example_ids.append(entry['example_id'])

            entry_ids[i, :entry['entry_length']] = entry['entry_ids']
            entry_mask[i, :entry['entry_length']] = torch.ones(entry['entry_length'], dtype=torch.long)
            


        

        batch_entry['input_ids'] = input_ids
        batch_entry['input_types'] = input_types
        batch_entry['label'] = label_ids
        batch_entry['text_id'] = text_ids
        batch_entry['example_id'] = example_ids
        batch_entry['input_mask'] = input_mask
        

        batch_entry['entry_ids'] = entry_ids
        batch_entry['entry_mask'] = entry_mask

        return batch_entry


    def get_labels(self, ):
        if self.args.task_name == 'chid':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            raise ValueError('INVALID TASK')




def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)
    if args.debug:
        verbose = True

    dataset = CHIDDataset(
        split,
        verbose=verbose,
        args=args,
        mode=mode
        )

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


    loader.task = args.task_name
    if verbose:
        loader.evaluator = TCEvaluator(dataset)

    return loader, len(dataset.get_labels())


class TCEvaluator: # for wandb evaluate
    def __init__(self, dataset):
        self.dataset = dataset
    def evaluate(self, textid2ans: dict):
        # score = 0
        # for ans_item in res:
        score = 0.
        for textid, ans in textid2ans.items():
            line = self.dataset.id2datum[textid]
            label = self.dataset.labels2id[line['label']]
            if ans == label:
                score += 1
        return score / len(textid2ans)

    def dump_result(self, textid2ans: dict, path):
        """
        for CLUE evaluation
        tnews_predict.json
        {'id':1, 'label':'102', 'label_desc':'news_entertainment'}
        """
        result = {}
        labels = self.dataset.get_labels()
        for text_id, ans in textid2ans.items():
            result[text_id] = int(labels[ans])
            # result.append({'id':text_id, 'label':int(labels[ans])})
        with open(path, 'w', encoding='utf8') as jh:
            json.dump(result, jh)
            # for item in result:
            #     jh.write(json.dumps(item))
            #     jh.write('\n')




        
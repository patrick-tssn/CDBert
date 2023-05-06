
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
CLUE_dir = dataset_dir.joinpath('KEY/')



class MRCDataset(Dataset):

    def __init__(self, split='train', verbose=True, args=None, max_length: int = 512, mode='train'):
        super().__init__()
        
        
        self.max_length = max_length
        self.max_text_length = args.max_text_length
        self.args = args
        with open(dataset_dir.joinpath('KEY/{}/{}.json'.format(args.task_name, split)), 'r', encoding='utf8') as jh:
            lines = jh.readlines()
        print(args.task_name)
        # self.lines = lines
        # add text id
        self.lines = []
        self.id2datum = {}
        for idx, line in enumerate(lines):
            line = json.loads(line)

            if args.task_name == 'key': # context question answer
                answers = line['options']
                context = '\n'.join(line['sentence'])
                label = line['label']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'context':context, 'answers':answers, 'entrys':entrys})
                self.id2datum[str(idx)] = {'label':label, 'context':context, 'answer':answers, 'entrys':entrys}
            
           

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
        text_idx,label,context,answers,entrys = line['text_idx'],line['label'],line['context'],line['answers'],line['entrys']
        # entrys
        label = self.labels2id[label]
        # sentence = sentence[:self.max_seq-2]
        # convert characters to input ids
        context_tokens = self.tokenizer.tokenize(context)
        input_token_ids = []
        input_token_types = []
        input_attention_masks = []
        input_entry_hidden_states = []
        for answer in answers:
            answer_tokens = self.tokenizer.tokenize(answer)
            context_tokens,  answer_tokens = self.truncate_seq_tuple(context_tokens[:], answer_tokens[:], self.args.max_text_length-4)
            src_a = ['[CLS]'] + answer_tokens + ['[SEP]']
            src_b = context_tokens + ['[SEP]']
            
            seg_a = [0] * len(src_a)
            seg_b = [1] * len(src_b)
            bert_tokens = self.tokenizer.convert_tokens_to_ids(src_a + src_b)
            # token_type_ids = self.tokenizer.converst_tokens_to_ids(seg_a + seg_b)
            token_type_ids = seg_a + seg_b
            attention_mask_ids = [1] * len(token_type_ids)
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
        desc_batch_ids = []
        for i, entry in enumerate(batch):
            input_ids[i] = entry['input_ids']
            input_types[i] = entry['input_types']
            input_mask[i] = entry['input_masks']
            label_ids[i, 0] = entry['label']
            # clip_embeds[i, :entry['input_length']] = entry['clip_embeds']
            text_ids.append(entry['text_id'])

            entry_ids[i, :entry['entry_length']] = entry['entry_ids']
            entry_mask[i, :entry['entry_length']] = torch.ones(entry['entry_length'], dtype=torch.long)

        

        batch_entry['input_ids'] = input_ids
        batch_entry['input_types'] = input_types
        batch_entry['label'] = label_ids
        batch_entry['text_id'] = text_ids
        batch_entry['input_mask'] = input_mask
        

        batch_entry['entry_ids'] = entry_ids
        batch_entry['entry_mask'] = entry_mask

        return batch_entry


    def get_labels(self, ):
        if self.args.task_name == 'key':
            return [0, 1, 2, 3]
        else:
            raise ValueError('INVALID TASK')


    def truncate_seq_tuple(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence tuple in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) >= len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)
    if args.debug:
        verbose = True

    dataset = MRCDataset(
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
        result = []
        labels = self.dataset.get_labels()
        for text_id, ans in textid2ans.items():
            result.append({'id':int(text_id), 'label':labels[ans]})
        with open(path, 'w', encoding='utf8') as jh:
            for item in result:
                jh.write(json.dumps(item))
                jh.write('\n')




        
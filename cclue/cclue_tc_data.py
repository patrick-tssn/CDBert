
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
CLUE_dir = dataset_dir.joinpath('CCLUE/')



class TCDataset(Dataset):

    def __init__(self, split='train', verbose=True, args=None, max_length: int = 512, mode='train'):
        super().__init__()
        
        
        self.max_length = max_length
        self.max_text_length = args.max_text_length
        self.args = args
        with open(dataset_dir.joinpath('CCLUE/{}/{}.json'.format(args.task_name, split)), 'r', encoding='utf8') as jh:
            lines = jh.readlines()
        print(args.task_name)
        # self.lines = lines
        # add text id
        self.lines = []
        self.id2datum = {}
        for idx, line in enumerate(lines):
            line = json.loads(line)

            if args.task_name == 'tc':
                label = line['label']
                sent = line['sentence']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys}
            elif args.task_name == 'fspc':
                label = line['label']
                sent = line['sentence']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys}
            else:
                raise ValueError('INVALID TASK NAME')
            
            
            

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
        text_idx,label,sentence1,sentence2,entrys = line['text_idx'],line['label'],line['sentence1'],line['sentence2'],line['entrys']
        # entrys
        label = self.labels2id[label]
        # sentence = sentence[:self.max_seq-2]
        # convert characters to input ids
        # bert_tokens = self.tokenizer.encode(sentence, )
        # bert_tokens = self.tokenizer.encode(sentence, padding=True, truncation=True, max_length=self.max_text_length)
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.args.max_text_length
        )
        bert_tokens, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # convert token_id to embeds
        glyph_ids = []
        entry_ids = []
        desc_ids = []
        entry_radical_ids = []
        desc_ids = []
        desc_radical_ids = []
        desc_idx = [0]

        for ii, entry in enumerate(entrys):
            
            # desc_ids += self.hs_dict[entry]
            # desc_idx.append(desc_idx[-1] + len(self.hs_dict[entry]))
            desc_ids.append(self.entry_dict[entry])
            




        # bert_tokens = tokenizer_output.ids
        # pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        # assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        input_types = torch.LongTensor(token_type_ids)
        # pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(label)])

        # if desc_ids != []:
        #     desc_ids = torch.cat(desc_ids, dim=0)

        # input_token_type_ids = torch.ones(input_ids.size(), dtype=torch.LongTensor) * 0
        
        out_dict = {}
        out_dict['input_ids'] = input_ids
        out_dict['input_types'] = input_types
        out_dict['input_length'] = len(input_ids)
        out_dict['label'] = label
        out_dict['text_id'] = text_idx

        if desc_ids != []:
            out_dict['entry_ids'] = torch.cat(desc_ids)[:self.args.max_text_length]
            out_dict['entry_length'] = len(out_dict['entry_ids'])
        else:
            out_dict['entry_ids'] = torch.zeros(1, self.hd_sz)
            out_dict['entry_length'] = 0

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        # args = batch[0]['args']
        B = len(batch)
        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_types = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        label_ids = torch.ones(B, 1, dtype=torch.long)
        input_mask = torch.zeros(B, S_W_L, dtype=torch.long)
        
        
        E_W_L = max(entry['entry_length'] for entry in batch)
        entry_ids = torch.zeros((B, E_W_L, self.hd_sz), dtype=torch.float)
        entry_mask = torch.zeros(B, E_W_L, dtype=torch.long)


        text_ids = []
        glyph_idx = 0
        entry_idx = 0
        desc_idx = 0
        desc_batch_ids = []
        desc_batch_idx = []
        batch_idx = [0]
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_types[i, :entry['input_length']] = entry['input_types']
            label_ids[i, 0] = entry['label']
            input_mask[i, :entry['input_length']] = torch.ones(entry['input_length'], dtype=torch.long)
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
        if self.args.task_name == 'tc':
            return ['101','102','103','104', '105', '106','107','108','109','110']
        elif self.args.task_name == 'fspc':
            return ['1', '2', '3', '4', '5']
        else:
            raise ValueError('INVALID TASK')

    def get_label_desc(self, ):
        if self.args.task_name == 'tc':
            return ['易藏', '医藏', '艺藏', '史藏', '佛藏', '集藏', '诗藏', '子藏', '儒藏', '道藏']
        elif self.args.task_name == 'fspc':
            return ['1', '2', '3', '4', '5']
        else:
            raise ValueError('INVALID TASK')


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)
    if args.debug:
        verbose = True

    dataset = TCDataset(
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
        for CCLUE evaluation
        tnews_predict.json
        {'id':1, 'label':'102', 'label_desc':'news_entertainment'}
        """
        result = []
        labels = self.dataset.get_labels()
        label_descs = self.dataset.get_label_desc()
        for text_id, ans in textid2ans.items():
            result.append({'id':int(text_id), 'label':labels[ans], 'label_desc':label_descs[ans]})
        with open(path, 'w', encoding='utf8') as jh:
            for item in result:
                jh.write(json.dumps(item))
                jh.write('\n')




        
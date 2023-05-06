
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

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizerFast


project_dir = Path(__file__).resolve().parent.parent  
workspace_dir = project_dir
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
CLUE_dir = dataset_dir.joinpath('CCLUE/')



class SLDataset(Dataset):

    def __init__(self, split='train', verbose=True, args=None, max_length: int = 512, mode='train'):
        super().__init__()
        
        
        self.max_length = max_length
        self.max_text_length = args.max_text_length
        self.args = args
        with open(dataset_dir.joinpath('CCLUE/{}/{}.json'.format(args.task_name, split)), 'r', encoding='utf8') as jh:
            lines = jh.readlines()
        print(args.task_name)
        labels = self.get_labels()
        self.labels2id = {value: key for key, value in enumerate(labels)}
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.args.backbone,
        )
        # self.lines = lines
        # add text id
        self.lines = []
        self.id2datum = {}
        for idx, line in enumerate(lines):
            line = json.loads(line)

            if args.task_name == 'ner':
                label = line['ner_tags']
            elif args.task_name == 'punc':
                label = line['punc_tags']
            elif args.task_name == 'seg':
                label = line['seg_tags']
            else:
                raise ValueError('INVALID TASK NAME: ', args.task_name)
            tokens = line['tokens']
            entrys = line['entrys']
            tokenized_inputs = self.tokenizer(
                tokens,
                padding=False, # 'max_length'
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            bert_tokens = tokenized_inputs['input_ids']
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.labels2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(self.labels2id[label[word_idx]])
                previous_word_idx = word_idx
            self.lines.append({'text_idx':str(idx), 'label': label_ids, 'tokens':bert_tokens, 'entrys':entrys})
            self.id2datum[str(idx)] = {'label':label_ids, 'tokens':bert_tokens, 'entrys':entrys}
            
            
    
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
        text_idx,label,bert_tokens,entrys = line['text_idx'],line['label'],line['tokens'],line['entrys']
        
            
        
        

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
            
            # glyph_id = []
            # for character in entry:
            #     idx = self.tokenizer.convert_tokens_to_ids(character)
            #     glyph_id.append(torch.from_numpy(self.glyph_feature[str(idx)][()]).view(1, -1))
            # entry_ids.append(torch.LongTensor(self.tokenizer.encode(entry)))
            # entry_radical_ids.append(torch.LongTensor(self.dict[entry]))
            # glyph_ids.append(torch.cat(glyph_id, dim=0))
            # # descs = [desc.split('。')[0].split('：')[0][:30] for desc in self.word_desc[entry]]
            
            # descs = [desc[:30] for desc in self.word_desc[entry]]
            # desc_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'+''.join(descs)))))
            # desc_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'+descs[ii]))))

            # for jj, desc in enumerate(self.dict[entry]['desc']):
                
            #     desc_ids.append(torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entry + '[SEP]' + desc))))
            #     desc_radical_ids.append(torch.LongTensor(self.dict[entry]['entry_radical'][jj] + [280] + self.dict[entry]['desc_radical'][jj]))
            # desc_idx.append(desc_idx[-1]+jj+1)
            
        # glyph_embeds = torch.cat(glyph_ids, dim=0)
        # clip_embeds = torch.zeros(len(bert_tokens), 512)
        # for ii, tk in enumerate(bert_tokens):
        #     clip_embeds[ii] = torch.tensor(self.char2clip[str(tk)])



        # bert_tokens = tokenizer_output.ids
        # pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        # assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        # input_types = torch.LongTensor(token_type_ids)
        # pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor(label)

        # if desc_ids != []:
        #     desc_ids = torch.cat(desc_ids, dim=0)

        # input_token_type_ids = torch.ones(input_ids.size(), dtype=torch.LongTensor) * 0
        
        out_dict = {}
        out_dict['input_ids'] = input_ids
        # out_dict['input_types'] = input_types
        out_dict['input_length'] = len(input_ids)
        out_dict['label'] = label
        out_dict['text_id'] = text_idx
        
        # out_dict['glyph_embeds'] = glyph_ids
        # out_dict['glyph_length'] = len(glyph_ids)
        # out_dict['entry_ids'] = entry_ids
        # out_dict['entry_length'] = len(entry_ids)
        # if desc_ids != []:
        #     out_dict['input_desc_ids'] = torch.cat((input_ids, desc_ids)) 
        #     out_dict['input_token_type_ids'] = torch.cat((torch.zeros(input_ids.size(), dtype=torch.long), torch.ones(desc_ids.size(), dtype=torch.long)))
        #     out_dict['input_desc_length'] = len(out_dict['input_desc_ids'])
        # else:
        #     out_dict['input_desc_ids'] = input_ids
        #     out_dict['input_token_type_ids'] = torch.zeros(input_ids.size(), dtype=torch.long)
        #     out_dict['input_desc_length'] = len(out_dict['input_desc_ids'])
        # out_dict['clip_embeds'] = clip_embeds
        if desc_ids != []:
            out_dict['entry_ids'] = torch.cat(desc_ids)[:self.args.max_text_length]
            out_dict['entry_length'] = len(out_dict['entry_ids'])
        else:
            out_dict['entry_ids'] = torch.zeros(1, self.hd_sz)
            out_dict['entry_length'] = 0
            # print(text_idx)
        # out_dict['desc_idx'] = desc_idx
        # out_dict['desc_radical_ids'] = desc_radical_ids
        # out_dict['desc_idx'] = desc_idx
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        # args = batch[0]['args']
        B = len(batch)
        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # input_types = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        label_ids = torch.ones(B, S_W_L, dtype=torch.long)
        input_mask = torch.zeros(B, S_W_L, dtype=torch.long)
        
        
        E_W_L = max(entry['entry_length'] for entry in batch)
        entry_ids = torch.zeros((B, E_W_L, self.hd_sz), dtype=torch.float)
        entry_mask = torch.zeros(B, E_W_L, dtype=torch.long)
        # entry_seg = [len(entrys['entry_ids']) for entrys in batch]
        # EB = sum(entry_seg)
        # E_W_L = max(entry.size(0) for entrys in batch for entry in entrys['entry_ids'])
        # G_W_L = max(entry.size(0) for entrys in batch for entry in entrys['glyph_embeds'])
        # entry_ids = torch.ones(EB, E_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # glyph_embeds = torch.zeros((EB, G_W_L, 512), dtype=torch.float32)
        # glyph_mask_ids = torch.zeros(EB, G_W_L, dtype=torch.long)
        
        # DESC_W_L = min(max(entry['input_desc_length'] for entry in batch), self.max_length-2)
        # input_desc_ids = torch.ones(B, DESC_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # input_desc_token_type_ids = torch.ones(B, DESC_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # input_desc_mask = torch.zeros(B, DESC_W_L, dtype=torch.long)
        
        # DB = sum(x['desc_ids'].size(0) for x in batch)
        # desc_embs = torch.ones(DB, 768, dtype=torch.float)
        # D_W_L = max(len(x) for entry in batch for x in entry['desc_ids'])
        # desc_ids = torch.ones(DB, D_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # desc_radical_ids = torch.ones(DB, D_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # desc_input_mask = torch.zeros(DB, D_W_L, dtype=torch.long)

        text_ids = []
        glyph_idx = 0
        entry_idx = 0
        desc_idx = 0
        desc_batch_ids = []
        desc_batch_idx = []
        batch_idx = [0]
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            # input_types[i, :entry['input_length']] = entry['input_types']
            label_ids[i, :entry['input_length']] = entry['label']
            input_mask[i, :entry['input_length']] = torch.ones(entry['input_length'], dtype=torch.long)
            # clip_embeds[i, :entry['input_length']] = entry['clip_embeds']
            text_ids.append(entry['text_id'])

            # for ent in entry['glyph_embeds']:
            #     glyph_embeds[glyph_idx, :ent.size(0)] = ent
            #     glyph_mask_ids[glyph_idx, :ent.size(0)] = torch.ones(ent.size(0), dtype=torch.long)
            #     glyph_idx += 1
            # # glyph_embeds[i, :entry['glyph_length']] = entry['glyph_embeds']
            # # glyph_mask_ids[i, :entry['glyph_length']] = torch.ones(entry['glyph_length'], dtype=torch.long)
            # for ent in entry['entry_ids']:
            #     entry_ids[entry_idx, :ent.size(0)] = ent
            #     entry_idx += 1
            
            # for jj, desc in enumerate(entry['desc_ids']):
            #     desc_ids[desc_idx, :len(desc)] = desc
            #     desc_radical_ids[desc_idx, :len(desc)] = entry['desc_radical_ids'][jj]
            #     desc_input_mask[desc_idx, :len(desc)] = torch.ones(len(desc), dtype=torch.long)
            #     desc_idx += 1
            entry_ids[i, :entry['entry_length']] = entry['entry_ids']
            entry_mask[i, :entry['entry_length']] = torch.ones(entry['entry_length'], dtype=torch.long)
            # desc_batch_idx.append(entry['desc_idx'])
            # batch_idx.append(batch_idx[-1] + entry['desc_idx'][-1])
                
            # input_desc_ids[i, :entry['input_desc_length']] = entry['input_desc_ids'][:min(entry['input_desc_length'], DESC_W_L)]
            # input_desc_token_type_ids[i, :entry['input_desc_length']] = entry['input_token_type_ids'][:min(entry['input_desc_length'], DESC_W_L)]
            # input_desc_mask[i, :entry['input_desc_length']] = torch.ones(min(entry['input_desc_length'], DESC_W_L), dtype=torch.long)
        # desc_embs = torch.cat(desc_batch_ids)
        

        batch_entry['input_ids'] = input_ids
        # batch_entry['input_types'] = input_types
        batch_entry['label'] = label_ids
        batch_entry['text_id'] = text_ids
        batch_entry['input_mask'] = input_mask
        
        # # batch_entry['clip_embeds'] = clip_embeds
        # batch_entry['glyph_embeds'] = glyph_embeds
        # batch_entry['glyph_mask_ids'] = glyph_mask_ids
        # batch_entry['entry_ids'] = entry_ids
        # batch_entry['entry_segs'] = entry_seg
        
        # batch_entry['input_desc_ids'] = input_desc_ids
        # batch_entry['input_desc_mask'] = input_desc_mask
        # batch_entry['input_desc_token_type_ids'] = input_desc_token_type_ids
        # batch_entry['desc_ids'] = desc_ids
        # batch_entry['desc_radical_ids'] = desc_radical_ids
        # batch_entry['desc_input_mask'] = desc_input_mask
        batch_entry['entry_ids'] = entry_ids
        batch_entry['entry_mask'] = entry_mask
        # batch_entry['desc_idx'] = desc_batch_idx
        # batch_entry['batch_idx'] = batch_idx
        return batch_entry


    def get_labels(self, ):
        if self.args.task_name == 'ner':
            return ['O', 'B-NOUN_BOOKNAME', 'I-NOUN_BOOKNAME', 'B-NOUN_OTHER', 'I-NOUN_OTHER']
        elif self.args.task_name == 'punc':
            return ['O', 'B-,', 'B-.', 'B-?', 'B-!', 'B-\\', 'B-:', 'B-;']
        elif self.args.task_name == 'seg':
            return ['O', 'B']
        else:
            raise ValueError('INVALID TASK')

def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)
    if args.debug:
        verbose = True

    dataset = SLDataset(
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
        loader.evaluator = SLEvaluator(dataset)

    return loader, len(dataset.get_labels())


class SLEvaluator: # for wandb evaluate
    def __init__(self, dataset):
        self.dataset = dataset
    def evaluate(self, textid2ans: dict):
        # score = 0
        # for ans_item in res:
        score = 0.
        true_predictions = []
        true_labels = []
        label_list = self.dataset.get_labels()
        for textid, ans in textid2ans.items():
            line = self.dataset.id2datum[textid]
            label = line['label']
            true_predictions.append([label_list[p] for (p, l) in zip(ans, label) if l != -100])
            true_labels.append([label_list[l] for (p, l) in zip(ans, label) if l != -100])
        
        acc = accuracy_score(true_labels, true_predictions)
        prec = precision_score(true_labels, true_predictions)
        recl = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        
        return acc, prec, recl, f1

    def dump_result(self, textid2ans: dict, path):
        """
        for CCLUE evaluation
        tnews_predict.json
        {'id':1, 'label':'102', 'label_desc':'news_entertainment'}
        """
        result = []
        true_predictions = []
        true_labels = []
        label_list = self.dataset.get_labels()
        for textid, ans in textid2ans.items():
            line = self.dataset.id2datum[textid]
            label = line['label']
            result.append({'id':int(textid), 'label':[label_list[p] for (p, l) in zip(ans, label) if l != -100]})
            
        with open(path, 'w', encoding='utf8') as jh:
            for item in result:
                jh.write(json.dumps(item))
                jh.write('\n')




        
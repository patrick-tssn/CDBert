
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
import nltk
import jieba

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


from transformers import BertTokenizer, BertTokenizerFast
import transformers
transformers.logging.set_verbosity_error()

CLS_TOKEN, PAD_TOKEN, SEP_TOKEN = '[CLS]', '[SEP]', '[SEP]'

project_dir = Path(__file__).resolve().parent.parent 
workspace_dir = project_dir
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
CLUE_dir = dataset_dir.joinpath('CLUE/')


class QADataset(Dataset):

    def __init__(self, split='train', verbose=True, args=None, max_length: int = 512, mode='train'):
        super().__init__()
        
        with open(dataset_dir.joinpath('zhs/pretrain_char_word_example_vocab.json')) as jh:
            word_dict = json.load(jh)
        dict_set = set(word_dict)
        vocab_path = str(dataset_dir.joinpath('zhs/jieba_char_word_vocab.txt'))
        if not os.path.exists(vocab_path):
            with open(vocab_path, 'w') as fh:
                for word in dict_set:
                    fh.writelines(word)
                    fh.write('\n')
        jieba.load_userdict(vocab_path)
        
        self.max_length = max_length
        self.max_text_length = args.max_text_length
        self.args = args
        with open(dataset_dir.joinpath('CLUE/{}/{}.json'.format(args.task_name, split)), 'r', encoding='utf8') as jh:
            lines = jh.readlines()
        print(args.task_name)
        
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.args.backbone,
        )
        
        # convert examples to dataset
        # examples -> data item
        # add text id
        self.lines = []
        self.id2datum = {}
        
        
        
        if verbose:
            print('start preprocess examples')
        for line in tqdm(lines):
            line = json.loads(line)

            if args.task_name == 'cmrc':
                if 'test' in split: 
                    answers = ['无效答案']
                    answer_starts = [0]
                else: 
                    answers = line['answers']
                    answer_starts = line['answer_starts']
                idx = line['idx']
                question = line['question']
                context = line['context']
                
                # mapping token position to character position
                # method1 default tokenizer
                context_tokens_tk = self.tokenizer(context, return_offsets_mapping=True)
                context_tokens = context_tokens_tk['input_ids'][1:-1]
                context_spans = context_tokens_tk['offset_mapping'][1:-1]

                
                question_tokens = self.tokenizer.tokenize(question)
                
                self.id2datum[idx] = {'answer':answers, 'label':answer_starts, 'question':question, 'context':context, 'context_spans':context_spans}
                
  
                
                # trick 2: fine position
                answer_tokens = self.tokenizer.encode(answers[0])[1:-1]
                n_answer = len(answer_tokens)
                for ii in range(len(context_tokens)):
                    if context_tokens[ii:ii+n_answer] == answer_tokens:
                        start_position_absolute = ii
                        end_position_absolute = ii + n_answer - 1
                        break
                else:
                    start_position_absolute = answer_starts[0]
                    end_position_absolute = answer_starts[0] + len(answers[0])  - 1
                    for i, span in enumerate(context_spans):
                        if start_position_absolute >= span[0] and start_position_absolute < span[1]:
                            start_position_absolute = i
                            break
                        if span[0] > start_position_absolute:
                            start_position_absolute = i
                            break
                    for i, span in enumerate(context_spans):
                        if end_position_absolute >= span[0] and end_position_absolute < span[1]:
                            end_position_absolute = i
                            break
                        if span[0] > end_position_absolute:
                            end_position_absolute = i-1
                            break
                    if 'train' in split:
                        print('blur answer: ', answers[0], '\n', 'context: ', context)
                
                    
                
                # cut context by doc_stride(default: 256)
                
                max_context_length = self.max_text_length - len(question_tokens) - 3
                doc_spans = []
                start_offset = 0
                while start_offset < len(context_tokens):
                    length = len(context_tokens) - start_offset
                    if length > max_context_length:
                        length = max_context_length
                    doc_spans.append((start_offset, length))
                    if start_offset + length == len(context_tokens):
                        break
                    start_offset += min(length, args.doc_stride)
                for doc_span_index, doc_span in enumerate(doc_spans):
                    start_offset, span_length = doc_span[0], doc_span[1]
                    span_context = context[context_spans[start_offset][0]: context_spans[start_offset+span_length-1][1]]
                    # span_context_tokens = context_tokens[start_offset: start_offset+doc_span[1]]
                    # 获得 context entry
                    entrys_tp = jieba.tokenize(question+span_context)
                    entrys, entry_spans = [], []
                    for ent in entrys_tp:
                        if ent[0] in dict_set:
                            entrys.append(ent[0])
                            entry_spans.append([ent[1], ent[2]])
                    # 获得相对位置
                    start_position = start_position_absolute - start_offset + len(question_tokens) + 2
                    end_position = end_position_absolute - start_offset + len(question_tokens) + 2
                    # 不完整的数据作为数据增强
                    if start_position < len(question_tokens) + 2:
                        start_position = len(question_tokens) + 2
                    if end_position > doc_span[1] + len(question_tokens) + 1:
                        end_position = doc_span[1] + len(question_tokens) + 1
                    if start_position > doc_span[1] + len(question_tokens) + 1 or end_position < len(question_tokens) + 2:
                        start_position, end_position = 0, 0
                        
                    # 获得 max_context_score，重复出现的词应该取中间的
                    token_is_max_context = {}
                    question_start_offset = len(question_tokens) + 2
                    for mc_i in range(span_length):
                        mc_pos = start_offset + mc_i
                        best_score = 0
                        best_span_index = 0
                        for mc_idx, mc_span in enumerate(doc_spans):
                            mc_start = mc_span[0]
                            mc_length = mc_span[1]
                            mc_end = mc_start+mc_length-1
                            if mc_pos < mc_start or mc_pos > mc_end: continue
                            mc_num_left = mc_pos - mc_start
                            mc_num_right = mc_end - mc_pos
                            mc_score = min(mc_num_left, mc_num_right) + 0.01 * mc_length
                            if mc_score > best_score:
                                best_score = mc_score
                                best_span_index = mc_idx
                        token_is_max_context[question_start_offset+mc_i] = best_span_index == doc_span_index
                        
                    
                    # character tokens
                    # a_tokens = []
                    # for q in question:
                    #     t = self.tokenizer.tokenize(q)
                    #     if t:
                    #         a_tokens.append(t[0])
                    #     else:
                    #         a_tokens.append('[UNK]')
                    # b_tokens = []
                    # for c in span_context:
                    #     t = self.tokenizer.tokenize(c)
                    #     if t:
                    #         b_tokens.append(t[0])
                    #     else:
                    #         b_tokens.append('[UNK]')
                        
                    # src_a = self.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + a_tokens + [SEP_TOKEN])
                    # src_b = self.tokenizer.convert_tokens_to_ids(b_tokens + [SEP_TOKEN])
                    # src = src_a + src_b
                    # seg = [0] * len(src_a) + [1] * len(src_b)
                    # PAD_ID = self.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                    # while len(src) < args.max_text_length:
                    #     src.append(PAD_ID)
                    #     seg.append(0)
                    inputs = self.tokenizer.encode_plus(
                        question,
                        span_context,
                        add_special_tokens=True,
                        max_length=self.args.max_text_length
                    )
                    src, seg = inputs["input_ids"], inputs["token_type_ids"]
                    
                    self.lines.append({'input_id':src, 'segment_id':seg, 'item_idx':idx, 'start_position': start_position, 'end_position': end_position, 
                                       'answer':answers, 'question_length':len(question_tokens), 'doc_span_index':doc_span_index, 'start_offset':start_offset, 'entrys':entrys, 
                                       'token_is_max_context':token_is_max_context, 'context_spans':context_spans})
                


        
    
        if 'large' in self.args.backbone:
            self.hd_sz = 1024
        else:
            self.hd_sz = 768

        start = time.time()

        
        self.entry_dict = torch.load(dataset_dir.joinpath(args.embedding_lookup_table + 'entry_lookup_embedding.pt'))

        print('load embedding data cost {}'.format(time.time()-start))

        print('features: ', len(self.lines))
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        bert_tokens, token_type_ids, item_idx, start_position, end_position, answer, ques_len, doc_span_index, start_offset, entrys, token_is_max_context, context_spans = \
            line['input_id'], line['segment_id'], line['item_idx'], line['start_position'], line['end_position'], line['answer'], \
                line['question_length'], line['doc_span_index'], line['start_offset'], line['entrys'], line['token_is_max_context'], line['context_spans']

        desc_ids = []

        for ii, entry in enumerate(entrys):
            
            desc_ids.append(self.entry_dict[entry])
            
      


        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        input_types = torch.LongTensor(token_type_ids)
        start_label = torch.LongTensor([int(start_position)])
        end_label = torch.LongTensor([int(end_position)])


        # input_token_type_ids = torch.ones(input_ids.size(), dtype=torch.LongTensor) * 0
        
        out_dict = {}
        out_dict['input_ids'] = input_ids
        out_dict['input_types'] = input_types
        out_dict['input_length'] = len(input_ids)
        out_dict['start_label'] = start_label
        out_dict['end_label'] = end_label
        out_dict['answer'] = answer
        out_dict['item_idx'] = item_idx
        out_dict['question_length'] = ques_len
        out_dict['span_index'] = doc_span_index
        out_dict['start_offset'] = start_offset
        out_dict['token_is_max_context'] = token_is_max_context
        out_dict['context_spans'] = context_spans
        

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
        S_W_L = self.args.max_text_length
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_types = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        start_label_ids = torch.ones(B, 1, dtype=torch.long)
        end_label_ids = torch.ones(B, 1, dtype=torch.long)
        input_mask = torch.zeros(B, S_W_L, dtype=torch.long)
        
        E_W_L = max(entry['entry_length'] for entry in batch)
        entry_ids = torch.zeros((B, E_W_L, self.hd_sz), dtype=torch.float)
        entry_mask = torch.zeros(B, E_W_L, dtype=torch.long)



        
        idxs = []
        ques_lens = []
        span_indexs = []
        start_offsets = []
        token_is_max_contexts = []
        context_spans = []
        questions = []
        answers = []
        contexts = []
        glyph_idx = 0
        
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_types[i, :entry['input_length']] = entry['input_types']
            input_mask[i, :entry['input_length']] = torch.ones(entry['input_length'], dtype=torch.long)
            
            start_label_ids[i, 0] = entry['start_label']
            end_label_ids[i, 0] = entry['end_label']
            
            entry_ids[i, :entry['entry_length']] = entry['entry_ids']
            entry_mask[i, :entry['entry_length']] = torch.ones(entry['entry_length'], dtype=torch.long)
            
            # questions.append(entry['question'])
            # answers.append(entry['answer'])
            # contexts.append(entry['context'])
            idxs.append(entry['item_idx'])
            ques_lens.append(entry['question_length'])
            span_indexs.append(entry['span_index'])
            start_offsets.append(entry['start_offset'])
            token_is_max_contexts.append(entry['token_is_max_context'])
            context_spans.append(entry['context_spans'])

            # for ent in entry['glyph_embeds']:
            #     glyph_embeds[glyph_idx, :ent.size(0)] = ent
            #     glyph_mask_ids[glyph_idx, :ent.size(0)] = torch.ones(ent.size(0), dtype=torch.long)
            #     glyph_idx += 1
            # # glyph_embeds[i, :entry['glyph_length']] = entry['glyph_embeds']
            # # glyph_mask_ids[i, :entry['glyph_length']] = torch.ones(entry['glyph_length'], dtype=torch.long)
            
        

        batch_entry['input_ids'] = input_ids
        batch_entry['input_types'] = input_types
        batch_entry['input_mask'] = input_mask
        batch_entry['start_labels'] = start_label_ids
        batch_entry['end_labels'] = end_label_ids
        
        batch_entry['entry_ids'] = entry_ids
        batch_entry['entry_mask'] = entry_mask
        
        batch_entry['item_idxs'] = idxs
        batch_entry['question_lengths'] = ques_lens
        batch_entry['span_indexs'] = span_indexs
        batch_entry['start_offsets'] = start_offsets
        batch_entry['token_is_max_contexts'] = token_is_max_contexts
        batch_entry['context_spans'] = context_spans
        # batch_entry['questions'] = questions
        # batch_entry['contexts'] = contexts
        
        return batch_entry



def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)
    if args.debug:
        verbose = True

    dataset = QADataset(
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
        loader.evaluator = QAEvaluator(dataset)

    return loader


class QAEvaluator: # for wandb evaluate
    def __init__(self, dataset):
        self.dataset = dataset
    
    def remove_punctuation(self, in_str):
        in_str = str(in_str).lower().strip()
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                '「', '」', '（', '）', '－', '～', '『', '』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)
    
    def find_lcs(self, s1, s2):
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax
    
    def mixed_segmentation(self, in_str, rm_punc=False):
        in_str = str(in_str).lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                '「', '」', '（', '）', '－', '～', '『', '』']
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
                if temp_str != "":
                    # ss = nltk.word_tokenize(temp_str)
                    ss = list(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        # handling last part
        if temp_str != "":
            # ss = nltk.word_tokenize(temp_str)
            ss = list(temp_str)
            segs_out.extend(ss)

        return segs_out

    def calc_f1_score(self, answers, prediction):
        f1_scores = []
        for ans in answers:
            ans_segs = self.mixed_segmentation(ans, rm_punc=True)
            prediction_segs = self.mixed_segmentation(prediction, rm_punc=True)
            lcs, lcs_len = self.find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision   = 1.0*lcs_len/len(prediction_segs)
            recall      = 1.0*lcs_len/len(ans_segs)
            f1          = (2*precision*recall)/(precision+recall)
            f1_scores.append(f1)
        return max(f1_scores)


    def calc_em_score(self, answers, prediction):
        em = 0
        for ans in answers:
            ans_ = self.remove_punctuation(ans)
            prediction_ = self.remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em

    
    
    def evaluate(self, idx2ans: dict):
        # ans: token list
        em_score = 0.
        f1_score = 0.
        for idx, ans in idx2ans.items():
            span_index, start_pred, end_pred, score = ans
            line = self.dataset.id2datum[idx]
            answers = line['answer']
            context = line['context']
            context_spans = line['context_spans']
            # print(context_spans)
            # print(start_pred, end_pred)
            prediction = context[context_spans[start_pred][0]: context_spans[end_pred][1]]
            # cal em
            em_score += self.calc_em_score(answers, prediction)
            # cal f1
            f1_score += self.calc_f1_score(answers, prediction)
        
        return em_score / len(idx2ans), f1_score / len(idx2ans)

    def dump_result(self, idx2ans: dict, path):
        """
        for CLUE evaluation
        qa_predict.json
        {
            "TEST_0_QUERY_0": "美国海军提康德罗加级",
            "TEST_0_QUERY_1": "第二艘",
            "TEST_0_QUERY_2": "1862年"
        }
        """
        result = {}
        # labels = self.dataset.get_labels()
        # label_descs = self.dataset.get_label_desc()
        for idx, ans in idx2ans.items():
            span_index, start_pred, end_pred, score = ans
            line = self.dataset.id2datum[idx]
            context = line['context']
            context_spans = line['context_spans']
            prediction = context[context_spans[start_pred][0]: context_spans[end_pred][1]]
            result[idx] = prediction
        with open(path, 'w', encoding='utf8') as jh:
            json.dump(result, jh, ensure_ascii=False, indent='\t')




        
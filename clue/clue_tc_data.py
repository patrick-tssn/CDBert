
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



class TCDataset(Dataset):

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

            if args.task_name == 'tnews':
                if 'test' in split: label = '101'
                else: label = line['label']
                sent = line['sentence']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys}
            elif args.task_name == 'iflytek':
                if 'test' in split: label = '0'
                else: label = line['label']
                sent = line['sentence']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys}
            
            elif args.task_name == 'afqmc':
                if 'test' in split: label = '0'
                else: label = line['label']
                sent1, sent2 = line['sentence1'], line['sentence2']
                entrys1, entrys2 = line['entrys1'], line['entrys2']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent1, 'sentence2':sent2, "entrys":entrys1 + entrys2})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent1, 'sentence2':sent2, 'entrys':entrys1 + entrys2}
            elif args.task_name == 'ocnli':
                if 'test' in split: label = 'neutral'
                else: label = line['label']
                sent1, sent2 = line['sentence1'], line['sentence2']
                entrys1, entrys2 = line['entrys1'], line['entrys2']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent1, 'sentence2':sent2, "entrys":entrys1 + entrys2})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent1, 'sentence2':sent2, 'entrys':entrys1 + entrys2} 
            elif args.task_name == 'cmnli':
                if 'test' in split and 'label' not in line: label = 'neutral'
                else: label = line['label']
                sent1, sent2 = line['sentence1'], line['sentence2']
                entrys1, entrys2 = line['entrys1'], line['entrys2']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent1, 'sentence2':sent2, "entrys":entrys1 + entrys2})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent1, 'sentence2':sent2, 'entrys':entrys1 + entrys2}
            
            elif args.task_name == 'csl':
                if 'test' in split: label = '0'
                else: label = line['label']
                sent1, sent2 = ' '.join(line['keyword']), line['abst']
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent1, 'sentence2':sent2, "entrys":entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent1, 'sentence2':sent2, 'entrys':entrys}
            elif args.task_name == 'wsc':
                if 'test' in split: label = 'true'
                else: label = line['label']
                text_a = line['text']
                text_a_list = list(text_a)
                target = line['target']
                query = target['span1_text']
                query_idx = target['span1_index']
                pronoun = target['span2_text']
                pronoun_idx = target['span2_index']
                assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
                assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
                if pronoun_idx > query_idx:
                    text_a_list.insert(query_idx, "_")
                    text_a_list.insert(query_idx + len(query) + 1, "_")
                    text_a_list.insert(pronoun_idx + 2, "[")
                    text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
                else:
                    text_a_list.insert(pronoun_idx, "[")
                    text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                    text_a_list.insert(query_idx + 2, "_")
                    text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
                text_a = "".join(text_a_list)
                sent = text_a
                entrys = line['entrys']
                self.lines.append({'text_idx':str(idx), 'label': label, 'sentence1':sent, 'sentence2':None, "entrys":entrys})
                self.id2datum[str(idx)] = {'label':label, 'sentence1':sent, 'sentence2':None, 'entrys':entrys}
            

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
            

        assert len(bert_tokens) <= self.max_length

        input_ids = torch.LongTensor(bert_tokens)
        input_types = torch.LongTensor(token_type_ids)
        # pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(label)])

        
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
        # batch_entry['desc_idx'] = desc_batch_idx
        # batch_entry['batch_idx'] = batch_idx
        return batch_entry


    def get_labels(self, ):
        if self.args.task_name == 'tnews':
            return ['100', '101','102','103','104', '106','107','108','109','110','112','113','114','115','116']
        elif self.args.task_name == 'afqmc':
            return ['0', '1']
        elif self.args.task_name == 'cmnli':
            return ['neutral', 'entailment', 'contradiction']
        elif self.args.task_name == 'csl':
            return ['0', '1']
        elif self.args.task_name == 'ocnli':
            return ['neutral', 'entailment', 'contradiction']
        elif self.args.task_name == 'wsc':
            return ['true', 'false']
        elif self.args.task_name == 'iflytek':
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118']
        else:
            raise ValueError('INVALID TASK')

    def get_label_desc(self, ):
        if self.args.task_name == 'tnews':
            return ['news_story', 'news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car', 'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_stock', 'news_agriculture', 'news_game']
        elif self.args.task_name == 'afqmc':
            return ['0', '1']
        elif self.args.task_name == 'cmnli':
            return ['neutral', 'entailment', 'contradiction']
        elif self.args.task_name == 'csl':
            return ['0', '1']
        elif self.args.task_name == 'ocnli':
            return ['neutral', 'entailment', 'contradiction']
        elif self.args.task_name == 'wsc':
            return ['true', 'false']
        elif self.args.task_name == 'iflytek':
            return ['打车', '地图导航', '免费WIFI', '租车', '同城服务', '快递物流', '婚庆', '家政', '公共交通', '政务', '社区服务', '薅羊毛', '魔幻', '仙侠', '卡牌', '飞行空战', '射击游戏', '休闲益智', '动作类', '体育竞技', '棋牌中心', '经营养成', '策略', 'MOBA', '辅助工具', '约会社交', '即时通讯', '工作社交', '论坛圈子', '婚恋社交', '情侣社交', '社交工具', '生活社交', '微博博客', '新闻', '漫画', '小说', '技术', '教辅', '问答交流', '搞笑', '杂志', '百科', '影视娱乐', '求职', '兼职', '视频', '短视频', '音乐', '直播', '电台', 'K歌', '成人', '中小学', '职考', '公务员', '英语', '视频教育', '高等教育', '成人教育', '艺术', '语言(非英语)', '旅游资讯', '综合预定', '民航', '铁路', '酒店', '行程管理', '民宿短租', '出国', '工具', '亲子儿童', '母婴', '驾校', '违章', '汽车咨询', '汽车交易', '日常养车', '行车辅助', '租房', '买房', '装修家居', '电子产品', '问诊挂号', '养生保健', '医疗服务', '减肥瘦身', '美妆美业', '菜谱', '餐饮店', '体育咨讯', '运动健身', '支付', '保险', '股票', '借贷', '理财', '彩票', '记账', '银行', '美颜', '影像剪辑', '摄影修图', '相机', '绘画', '二手', '电商', '团购', '外卖', '电影票务', '社区超市', '购物咨询', '笔记', '办公', '日程管理', '女性', '经营', '收款', '其他']
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
        for CLUE evaluation
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




        
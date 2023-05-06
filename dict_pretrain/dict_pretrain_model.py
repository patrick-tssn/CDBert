import numpy as nps

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dict_modeling_bert import BertForDictPretraining

class BertPretraining(BertForDictPretraining):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')
        self.glyph = config.glyph
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        attention_mask = batch['mask_ids'].to(device)
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            glyph_embeds = 'glyph'
        else:
            radical_ids = None
            glyph_embeds = None
        
        # glyph_embeds = batch['glyph_embeds'].to(device)
        # glyph_mask_ids = batch['glyph_mask_ids'].to(device)
        # vis_pos = batch['boxes'].to(device)
        # synonym
        # syno_input_ids = batch['syno_input_ids'].to(device)
        # syno_glyph_embeds = batch['syno_glyph_embeds'].to(device)
        # syno_glyph_mask_ids = batch['syno_glyph_mask_ids'].to(device)
        # antonym
        # anto_input_ids = batch['anto_input_ids'].to(device)
        # anto_glyph_embeds = batch['anto_glyph_embeds'].to(device)
        # anto_glyph_mask_ids = batch['anto_glyph_mask_ids'].to(device)

        loss_weights = batch['loss_weights'].to(device) # balance distribution of labels

        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            radical_ids = radical_ids,
            glyph_embeds=glyph_embeds,
            # glyph_mask_ids=glyph_mask_ids,
            # syno_input_ids=syno_input_ids,
            # glyph_embeds=syno_glyph_embeds,
            # glyph_mask_ids=syno_glyph_mask_ids,
            # anto_input_ids=anto_input_ids,
            # glyph_embeds=anto_glyph_embeds,
            # glyph_mask_ids=anto_glyph_mask_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        # lm_mask = batch['mask_ids'].to(device)
        lm_mask = lm_labels != -100
        # lm_mask = torch.cat((lm_mask, torch.zeros(batch['glyph_mask_ids'].size(), dtype=torch.long, device=device)), dim=1)

        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        loss = loss

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}

        results = {}
        
        results['loss'] = (loss*loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        loss_weights = batch['loss_weights'].to(device)
        attention_mask = batch['mask_ids'].to(device)
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            glyph_embeds = 'glyph'
        else:
            radical_ids = None
            glyph_embeds = None

        # glyph_embeds = batch['glyph_embeds'].to(device)
        # glyph_mask_ids = batch['glyph_mask_ids'].to(device)
        # # synonym
        # syno_input_ids = batch['syno_input_ids'].to(device)
        # syno_glyph_embeds = batch['syno_glyph_embeds'].to(device)
        # syno_glyph_mask_ids = batch['syno_glyph_mask_ids'].to(device)
        # # antonym
        # anto_input_ids = batch['anto_input_ids'].to(device)
        # anto_glyph_embeds = batch['anto_glyph_embeds'].to(device)
        # anto_glyph_mask_ids = batch['anto_glyph_mask_ids'].to(device)

        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            # glyph_mask_ids=glyph_mask_ids,
            # syno_input_ids=syno_input_ids,
            # glyph_embeds=syno_glyph_embeds,
            # glyph_mask_ids=syno_glyph_mask_ids,
            # anto_input_ids=anto_input_ids,
            # glyph_embeds=anto_glyph_embeds,
            # glyph_mask_ids=anto_glyph_mask_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        # lm_mask = torch.cat((lm_mask, torch.zeros(batch['glyph_mask_ids'].size(), dtype=torch.long, device=device)), dim=1)
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        # loss, cl_loss = output['loss']
        loss = output['loss']
        loss = loss.view(B, L)*lm_mask
        loss = loss.sum(dim=1)/lm_mask.sum(dim=1).clamp(min=1)

        loss = loss
        
        results = {}
        results['loss'] = (loss*loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task:0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}
        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1
        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        pass
        
        
class BertCLPretraining(BertForDictPretraining):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')
        self.glyph = config.glyph
        self.eg = config.eg
        self.eg_only = config.eg_only
        
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        entry_ids = batch['entry_ids'].to(device)
        input_mask = batch['mask_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            target_radical_ids = batch['target_radical_ids'].to(device)
            
            syno_radical_ids = batch['syno_radical_ids'].to(device)
            anto_radical_ids = batch['anto_radical_ids'].to(device)
            
            mean_radical_ids = batch['mean_radical_ids'].to(device)
            flatten_mean_radical_ids = mean_radical_ids.view(-1, mean_radical_ids.size(-1))
            
            glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            entry_glyph_embeds = None
            mean_glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = 'glyph'
            entry_glyph_embeds = 'glyph'
            syno_glyph_embeds = 'glyph'
            anto_glyph_embeds = 'glyph'
            mean_glyph_embeds = 'glyph'
        else:
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = None
            entry_glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            mean_glyph_embeds = None
        
        syno_input_ids = batch['syno_target_ids'].to(device)
        syno_mask = batch['syno_mask_ids'].to(device)
        syno_loss_weight = batch['syno_loss_weight'].to(device)
        
        
        anto_input_ids = batch['anto_target_ids'].to(device)
        anto_mask = batch['anto_mask_ids'].to(device)
        anto_loss_weight = batch['anto_loss_weight'].to(device)
        
        # example
        if self.eg:
            example_input_ids = batch['example_input_ids'].to(device)
            example_mask_ids = batch['example_mask_ids'].to(device)
            mean_input_ids = batch['mean_input_ids'].to(device)
            mean_mask_ids = batch['mean_mask_ids'].to(device)
            mean_label_mask_ids = batch['mean_label_mask_ids'].to(device)
            mean_label_ids = batch['mean_label_ids'].to(device)
            flatten_mean_input_ids = mean_input_ids.view(-1, mean_input_ids.size(-1))
            flatten_mean_mask_ids = mean_mask_ids.view(-1, mean_mask_ids.size(-1))
            eg_loss_weight = batch['eg_loss_weight'].to(device)
        else:
            example_input_ids = None
            example_mask_ids = None
            mean_input_ids = None
            mean_mask_ids = None
            mean_label_mask_ids = None
            mean_label_ids = None
            flatten_mean_input_ids = None
            flatten_mean_mask_ids = None
            eg_loss_weight = None

        loss_weights = batch['loss_weights'].to(device) # balance distribution of labels

        output = self(
            input_ids=input_ids,
            attention_mask=input_mask,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            labels=lm_labels,
            entry_input_ids=entry_ids,
            entry_radical_ids=target_radical_ids,
            entry_glyph_embeds=entry_glyph_embeds,
            syno_input_ids=syno_input_ids,
            syno_radical_ids=syno_radical_ids,
            syno_glyph_embeds=syno_glyph_embeds,
            syno_input_mask=syno_mask,
            anto_input_ids=anto_input_ids,
            anto_radical_ids=anto_radical_ids,
            anto_glyph_embeds=anto_glyph_embeds,
            anto_input_mask=anto_mask,
            example_input_ids=example_input_ids,
            example_mask_ids=example_mask_ids,
            mean_input_ids=mean_input_ids,
            flatten_mean_input_ids=flatten_mean_input_ids,
            mean_mask_ids=mean_label_mask_ids,
            flatten_mean_mask_ids=flatten_mean_mask_ids,
            mean_label_ids=mean_label_ids,
            flatten_mean_radical_ids=flatten_mean_radical_ids,
            mean_glyph_embeds=mean_glyph_embeds,
            return_dict=True
        )
        assert 'loss' in output
        
        if self.eg:
            loss, cl_loss, eg_loss = output['loss']
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
            eg_mask = mean_label_ids != -100
            eg_mask = eg_mask.float()
            B,L  = mean_label_ids.size()
            eg_loss = eg_loss.view(B, L) * eg_mask
            eg_loss = eg_loss.sum(dim=1) / eg_mask.sum(dim=1).clamp(min=1)
            eg_loss = eg_loss * eg_loss_weight
            lm_mask = lm_labels != -100
            lm_mask = lm_mask.float()
            B, L = lm_labels.size()
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
            loss = loss*0.4 + cl_loss*0.3 + eg_loss*0.3
        else:
            loss, cl_loss = output['loss']
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
            lm_mask = lm_labels != -100
            lm_mask = lm_mask.float()
            B, L = lm_labels.size()
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
            loss = loss*0.4 + cl_loss*0.6

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}

        results = {}
        results['loss'] = (loss*loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)
        results['cl_loss'] = cl_loss.detach().sum()
        results['cl_loss_count'] = len(loss)
        if self.eg:
            results['eg_loss'] = eg_loss.detach().sum()
            results['eg_loss_count'] = len(loss)

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        entry_ids = batch['entry_ids'].to(device)
        input_mask = batch['mask_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            target_radical_ids = batch['target_radical_ids'].to(device)
            
            syno_radical_ids = batch['syno_radical_ids'].to(device)
            anto_radical_ids = batch['anto_radical_ids'].to(device)
            
            mean_radical_ids = batch['mean_radical_ids'].to(device)
            flatten_mean_radical_ids = mean_radical_ids.view(-1, mean_radical_ids.size(-1))
            
            glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            entry_glyph_embeds = None
            mean_glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = 'glyph'
            entry_glyph_embeds = 'glyph'
            syno_glyph_embeds = 'glyph'
            anto_glyph_embeds = 'glyph'
            mean_glyph_embeds = 'glyph'
        else:
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = None
            entry_glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            mean_glyph_embeds = None
        
        syno_input_ids = batch['syno_target_ids'].to(device)
        syno_mask = batch['syno_mask_ids'].to(device)
        syno_loss_weight = batch['syno_loss_weight'].to(device)
        
        
        anto_input_ids = batch['anto_target_ids'].to(device)
        anto_mask = batch['anto_mask_ids'].to(device)
        anto_loss_weight = batch['anto_loss_weight'].to(device)
        
        # example
        if self.eg:
            example_input_ids = batch['example_input_ids'].to(device)
            example_mask_ids = batch['example_mask_ids'].to(device)
            mean_input_ids = batch['mean_input_ids'].to(device)
            mean_mask_ids = batch['mean_mask_ids'].to(device)
            mean_label_mask_ids = batch['mean_label_mask_ids'].to(device)
            mean_label_ids = batch['mean_label_ids'].to(device)
            flatten_mean_input_ids = mean_input_ids.view(-1, mean_input_ids.size(-1))
            flatten_mean_mask_ids = mean_mask_ids.view(-1, mean_mask_ids.size(-1))
            eg_loss_weight = batch['eg_loss_weight'].to(device)
        else:
            example_input_ids = None
            example_mask_ids = None
            mean_input_ids = None
            mean_mask_ids = None
            mean_label_mask_ids = None
            mean_label_ids = None
            flatten_mean_input_ids = None
            flatten_mean_mask_ids = None
            eg_loss_weight = None

        loss_weights = batch['loss_weights'].to(device) # balance distribution of labels

        output = self(
            input_ids=input_ids,
            attention_mask=input_mask,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            labels=lm_labels,
            entry_input_ids=entry_ids,
            entry_radical_ids=target_radical_ids,
            entry_glyph_embeds=entry_glyph_embeds,
            syno_input_ids=syno_input_ids,
            syno_radical_ids=syno_radical_ids,
            syno_glyph_embeds=syno_glyph_embeds,
            syno_input_mask=syno_mask,
            anto_input_ids=anto_input_ids,
            anto_radical_ids=anto_radical_ids,
            anto_glyph_embeds=anto_glyph_embeds,
            anto_input_mask=anto_mask,
            example_input_ids=example_input_ids,
            example_mask_ids=example_mask_ids,
            mean_input_ids=mean_input_ids,
            flatten_mean_input_ids=flatten_mean_input_ids,
            mean_mask_ids=mean_label_mask_ids,
            flatten_mean_mask_ids=flatten_mean_mask_ids,
            mean_label_ids=mean_label_ids,
            flatten_mean_radical_ids=flatten_mean_radical_ids,
            mean_glyph_embeds=mean_glyph_embeds,
            return_dict=True
        )
        assert 'loss' in output

        
        if self.eg:
            loss, cl_loss, eg_loss = output['loss']
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
            eg_mask = mean_label_ids != -100
            eg_mask = eg_mask.float()
            B,L  = mean_label_ids.size()
            eg_loss = eg_loss.view(B, L) * eg_mask
            eg_loss = eg_loss.sum(dim=1) / eg_mask.sum(dim=1).clamp(min=1)
            eg_loss = eg_loss * eg_loss_weight
            lm_mask = lm_labels != -100
            lm_mask = lm_mask.float()
            B, L = lm_labels.size()
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
            loss = loss*0.4 + cl_loss*0.3 + eg_loss*0.3
        else:
            loss, cl_loss = output['loss']
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
            lm_mask = lm_labels != -100
            lm_mask = lm_mask.float()
            B, L = lm_labels.size()
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
            loss = loss*0.4 + cl_loss*0.6
                
        results = {}
        results['loss'] = (loss*loss_weights).mean()
        results['cl_loss'] = cl_loss.detach().sum()
        results['cl_loss_count'] = len(loss)
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)
        if self.eg:
            results['eg_loss'] = eg_loss.detach().sum()
            results['eg_loss_count'] = len(loss)

        task_counts = {task:0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}
        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1
        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        pass
        
class BertEGPretraining(BertForDictPretraining):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')
        self.glyph = config.glyph
        self.eg = config.eg
        self.cl = config.cl
        self.eg_only = config.eg_only
        
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        entry_ids = batch['entry_ids'].to(device)
        input_mask = batch['mask_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        
        # syno_anto
        if self.cl :
            syno_input_ids = batch['syno_target_ids'].to(device)
            syno_mask = batch['syno_mask_ids'].to(device)
            syno_loss_weight = batch['syno_loss_weight'].to(device)
            
            
            anto_input_ids = batch['anto_target_ids'].to(device)
            anto_mask = batch['anto_mask_ids'].to(device)
            anto_loss_weight = batch['anto_loss_weight'].to(device)
        else:
            syno_input_ids = None
            syno_mask = None
            syno_loss_weight = None
            anto_input_ids = None
            anto_mask = None
            anto_loss_weight = None
        
        # example
        if self.eg:
            example_input_ids = batch['example_input_ids'].to(device)
            example_mask_ids = batch['example_mask_ids'].to(device)
            mean_input_ids = batch['mean_input_ids'].to(device)
            mean_mask_ids = batch['mean_mask_ids'].to(device)
            mean_label_mask_ids = batch['mean_label_mask_ids'].to(device)
            mean_label_ids = batch['mean_label_ids'].to(device)
            flatten_mean_input_ids = mean_input_ids.view(-1, mean_input_ids.size(-1))
            flatten_mean_mask_ids = mean_mask_ids.view(-1, mean_mask_ids.size(-1))
            eg_loss_weight = batch['eg_loss_weight'].to(device)
        else:
            example_input_ids = None
            example_mask_ids = None
            mean_input_ids = None
            mean_mask_ids = None
            mean_label_mask_ids = None
            mean_label_ids = None
            flatten_mean_input_ids = None
            flatten_mean_mask_ids = None
            eg_loss_weight = None
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            target_radical_ids = batch['target_radical_ids'].to(device)
            
            syno_radical_ids = batch['syno_radical_ids'].to(device)
            anto_radical_ids = batch['anto_radical_ids'].to(device)
            
            mean_radical_ids = batch['mean_radical_ids'].to(device)
            flatten_mean_radical_ids = mean_radical_ids.view(-1, mean_radical_ids.size(-1))
            
            glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            entry_glyph_embeds = None
            mean_glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = 'glyph'
            entry_glyph_embeds = 'glyph'
            syno_glyph_embeds = 'glyph'
            anto_glyph_embeds = 'glyph'
            mean_glyph_embeds = 'glyph'
        else:
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = None
            entry_glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            mean_glyph_embeds = None
        
        

        loss_weights = batch['loss_weights'].to(device) # balance distribution of labels

        output = self(
            input_ids=input_ids,
            attention_mask=input_mask,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            labels=lm_labels,
            entry_input_ids=entry_ids,
            entry_radical_ids=target_radical_ids,
            entry_glyph_embeds=entry_glyph_embeds,
            syno_input_ids=syno_input_ids,
            syno_radical_ids=syno_radical_ids,
            syno_glyph_embeds=syno_glyph_embeds,
            syno_input_mask=syno_mask,
            anto_input_ids=anto_input_ids,
            anto_radical_ids=anto_radical_ids,
            anto_glyph_embeds=anto_glyph_embeds,
            anto_input_mask=anto_mask,
            example_input_ids=example_input_ids,
            example_mask_ids=example_mask_ids,
            mean_input_ids=mean_input_ids,
            flatten_mean_input_ids=flatten_mean_input_ids,
            mean_mask_ids=mean_label_mask_ids,
            flatten_mean_mask_ids=flatten_mean_mask_ids,
            mean_label_ids=mean_label_ids,
            flatten_mean_radical_ids=flatten_mean_radical_ids,
            mean_glyph_embeds=mean_glyph_embeds,
            return_dict=True
        )
        assert 'loss' in output
        
        loss, cl_loss, eg_loss = output['loss']
        
        if cl_loss:
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
        if eg_loss:
            eg_mask = mean_label_ids != -100
            eg_mask = eg_mask.float()
            B,L  = mean_label_ids.size()
            eg_loss = eg_loss.view(B, L) * eg_mask
            eg_loss = eg_loss.sum(dim=1) / eg_mask.sum(dim=1).clamp(min=1)
            eg_loss = eg_loss * eg_loss_weight
            
        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
        
        if cl_loss and eg_loss:
            loss = loss*0.6 + cl_loss*0.2 + eg_loss*0.2
        elif cl_loss:
            loss = loss*0.4 + cl_loss*0.6
        elif eg_loss:
            loss = loss*0.4 + eg_loss*0.6
            

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}

        results = {}
        results['loss'] = (loss*loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)
        
        if self.eg:
            results['eg_loss'] = eg_loss.detach().sum()
            results['eg_loss_count'] = len(loss)
        if self.cl:
            results['cl_loss'] = cl_loss.detach().sum()
            results['cl_loss_count'] = len(loss)
            

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        entry_ids = batch['entry_ids'].to(device)
        input_mask = batch['mask_ids'].to(device)
        lm_labels = batch['target_ids'].to(device)
        
        # syno_anto
        if self.cl :
            syno_input_ids = batch['syno_target_ids'].to(device)
            syno_mask = batch['syno_mask_ids'].to(device)
            syno_loss_weight = batch['syno_loss_weight'].to(device)
            
            
            anto_input_ids = batch['anto_target_ids'].to(device)
            anto_mask = batch['anto_mask_ids'].to(device)
            anto_loss_weight = batch['anto_loss_weight'].to(device)
        else:
            syno_input_ids = None
            syno_mask = None
            syno_loss_weight = None
            anto_input_ids = None
            anto_mask = None
            anto_loss_weight = None
        
        # example
        if self.eg:
            example_input_ids = batch['example_input_ids'].to(device)
            example_mask_ids = batch['example_mask_ids'].to(device)
            mean_input_ids = batch['mean_input_ids'].to(device)
            mean_mask_ids = batch['mean_mask_ids'].to(device)
            mean_label_mask_ids = batch['mean_label_mask_ids'].to(device)
            mean_label_ids = batch['mean_label_ids'].to(device)
            flatten_mean_input_ids = mean_input_ids.view(-1, mean_input_ids.size(-1))
            flatten_mean_mask_ids = mean_mask_ids.view(-1, mean_mask_ids.size(-1))
            eg_loss_weight = batch['eg_loss_weight'].to(device)
        else:
            example_input_ids = None
            example_mask_ids = None
            mean_input_ids = None
            mean_mask_ids = None
            mean_label_mask_ids = None
            mean_label_ids = None
            flatten_mean_input_ids = None
            flatten_mean_mask_ids = None
            eg_loss_weight = None
        
        if self.glyph == 'radical':
            radical_ids = batch['radical_ids'].to(device)
            target_radical_ids = batch['target_radical_ids'].to(device)
            
            syno_radical_ids = batch['syno_radical_ids'].to(device)
            anto_radical_ids = batch['anto_radical_ids'].to(device)
            
            mean_radical_ids = batch['mean_radical_ids'].to(device)
            flatten_mean_radical_ids = mean_radical_ids.view(-1, mean_radical_ids.size(-1))
            
            glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            entry_glyph_embeds = None
            mean_glyph_embeds = None
        elif self.glyph == 'glyph':
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = 'glyph'
            entry_glyph_embeds = 'glyph'
            syno_glyph_embeds = 'glyph'
            anto_glyph_embeds = 'glyph'
            mean_glyph_embeds = 'glyph'
        else:
            radical_ids = None
            target_radical_ids = None
            syno_radical_ids = None
            anto_radical_ids = None
            flatten_mean_radical_ids = None
            
            glyph_embeds = None
            entry_glyph_embeds = None
            syno_glyph_embeds = None
            anto_glyph_embeds = None
            mean_glyph_embeds = None
        
        

        loss_weights = batch['loss_weights'].to(device) # balance distribution of labels

        output = self(
            input_ids=input_ids,
            attention_mask=input_mask,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            labels=lm_labels,
            entry_input_ids=entry_ids,
            entry_radical_ids=target_radical_ids,
            entry_glyph_embeds=entry_glyph_embeds,
            syno_input_ids=syno_input_ids,
            syno_radical_ids=syno_radical_ids,
            syno_glyph_embeds=syno_glyph_embeds,
            syno_input_mask=syno_mask,
            anto_input_ids=anto_input_ids,
            anto_radical_ids=anto_radical_ids,
            anto_glyph_embeds=anto_glyph_embeds,
            anto_input_mask=anto_mask,
            example_input_ids=example_input_ids,
            example_mask_ids=example_mask_ids,
            mean_input_ids=mean_input_ids,
            flatten_mean_input_ids=flatten_mean_input_ids,
            mean_mask_ids=mean_label_mask_ids,
            flatten_mean_mask_ids=flatten_mean_mask_ids,
            mean_label_ids=mean_label_ids,
            flatten_mean_radical_ids=flatten_mean_radical_ids,
            mean_glyph_embeds=mean_glyph_embeds,
            return_dict=True
        )
        assert 'loss' in output
        
        loss, cl_loss, eg_loss = output['loss']
        
        if cl_loss:
            cl_loss = cl_loss * syno_loss_weight * anto_loss_weight
        if eg_loss:
            eg_mask = mean_label_ids != -100
            eg_mask = eg_mask.float()
            B,L  = mean_label_ids.size()
            eg_loss = eg_loss.view(B, L) * eg_mask
            eg_loss = eg_loss.sum(dim=1) / eg_mask.sum(dim=1).clamp(min=1)
            eg_loss = eg_loss * eg_loss_weight
            
        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
        
        if cl_loss and eg_loss:
            loss = loss*0.6 + cl_loss*0.2 + eg_loss*0.2
        elif cl_loss:
            loss = loss*0.4 + cl_loss*0.6
        elif eg_loss:
            loss = loss*0.4 + eg_loss*0.6
            

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}

        results = {}
        results['loss'] = (loss*loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)
        
        if self.eg:
            results['eg_loss'] = eg_loss.detach().sum()
            results['eg_loss_count'] = len(loss)
        if self.cl:
            results['cl_loss'] = cl_loss.detach().sum()
            results['cl_loss_count'] = len(loss)


        task_counts = {task:0 for task in self.losses}
        task_loss = {task:0 for task in self.losses}
        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1
        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        pass
        

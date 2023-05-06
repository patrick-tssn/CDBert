from pathlib import Path
import numpy as np
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertForSequenceClassification, BertModel, BertPreTrainedModel, BertLayer
from transformers.models.bert.modeling_bert import BertAttention, BertPooler
from transformers.modeling_utils import apply_chunking_to_forward
from transformers import BertTokenizer
from urllib3 import encode_multipart_formdata

from models.dict_modeling_bert import BertClipModel

class DictBlock(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.crossattention = BertAttention(config, position_embedding_type="absolute")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        self_attn=True,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if encoder_hidden_states is not None:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            if self_attn:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_past_key_value,
                    output_attentions,
                )
            else:
                cross_attention_outputs = self.crossattention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_past_key_value,
                    output_attentions,
                )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Attention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(hidden_states)
            value_layer = self.transpose_for_scores(hidden_states)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(hidden_states)
            value_layer = self.transpose_for_scores(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



class BertConcatPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertMeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        
        # mean pooling
        avg_token_tensor = hidden_states.mean(1)
        
        # logsumexp: seq_len * hd_sz -> hd_sz
        # avg_token_tensor = torch.logsumexp(hidden_states, dim=1)
        pooled_output = self.dense(avg_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertClipPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        
        # # mean pooling
        # avg_token_tensor = hidden_states.mean(1)
        
        # logsumexp: seq_len * hd_sz -> hd_sz
        avg_token_tensor = torch.logsumexp(hidden_states, dim=1)
        pooled_output = self.dense(avg_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertDictTC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dict_bert = BertClipModel(config)

        # concat task
        self.pooler = BertPooler(config)
        self.mean_pooler = BertMeanPooler(config)
        self.logsumexp_pooler = BertClipPooler(config)
        self.concat_pooler = BertConcatPooler(config)
        self.proj = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.num_labels)
        self.bert_classifier = nn.Linear(config.hidden_size, self.num_labels)

        # fuse
        self.dict_block = DictBlock(config)
        self.attention = Attention(config)
        self.pad_token_id = config.pad_token_id
        self.config = config
        
        self.fuse = config.fuse

        # Initialize weights and apply final processing
        self.post_init()
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        input_types = batch['input_types'].to(device)
        # glyph_embeds = batch['glyph_embeds'].to(device)
        # glyph_mask_ids = batch['glyph_mask_ids'].to(device)
        # entry_ids = batch['entry_ids'].to(device)
        # seg_idx = batch['entry_segs']
        input_mask = batch['input_mask'].to(device)
        B = input_ids.size(0)
        label = batch['label'].to(device)
        # clip_embeds = batch['clip_embeds'].to(device)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=input_types,
            # glyph_embeds=glyph_embeds,
            # glyph_mask_ids=glyph_mask_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        # desc_ids = batch['desc_ids'].to(device)
        # desc_radical_ids = batch['desc_radical_ids'].to(device)
        # desc_input_mask = batch['desc_input_mask'].to(device)
        
        # desc_bz = desc_ids.size()[0]
        # ck_bzs = math.ceil(desc_bz / 1000)
        
        # bert_logits_lst = []
        # for bz in range(ck_bzs):
        #     start = bz * 1000
        #     end = min(start+1000, desc_bz)
        #     with torch.no_grad():
        #         self.dict_bert.eval()
        #         dict_output = self.dict_bert(
        #             input_ids=desc_ids[start:end],
        #             radical_ids=desc_radical_ids[start:end],
        #             attention_mask=desc_input_mask[start:end],
        #             # glyph_embeds=glyph_embeds,
        #             # glyph_mask_ids=glyph_mask_ids,
        #             return_dict=True,
        #         )
        #         bert_logits_lst.append(dict_output[0])
            
        # retrieval
        
        # desc_idx = batch['desc_idx']
        # dict_logits = dict_output[0]
        # dict_logits = torch.cat(bert_logits_lst)
        desc_ids = batch['entry_ids']
        # batch_idx = batch['batch_idx']
        # desc_idx = batch['desc_idx']
        bert_logits = bert_output[0]
        bert_pooled_logits = bert_output[1]
        
        entry_ids = batch['entry_ids'].to(device)
        entry_mask = batch['entry_mask'].to(device)
        # batch_dict = []
        
        # idx = 0
        # for i in range(B):
        #     start, end = batch_idx[i], batch_idx[i+1]
        #     sim = F.cosine_similarity(bert_logits[i][0].expand(desc_ids[start:end].size()), desc_ids[start: end])
        #     retrieved_logits = []
        #     for j in range(len(desc_idx[i])-1):
        #         s, e = desc_idx[i][j], desc_idx[i][j+1]
        #         retrieved_logit = desc_ids[start:end][torch.argmax(sim[s:e])]
        #         if len(retrieved_logit.size()) != 0:
        #                 retrieved_logits.append(retrieved_logit)
        #     if retrieved_logits != []:
        #         batch_dict.append(torch.stack(retrieved_logits))
                
            
            
            # entry_idx_lst = desc_ids[i]
            # retrieved_logits = []
            # for entry_descs in entry_idx_lst:
            #     sim = []
            #     # each mean [CLS] + entry + desc : seq_len * 768
            #     for entry_desc in entry_descs:
            #         entry_desc = torch.FloatTensor(entry_desc).to(device)
            #         sim.append(F.cosine_similarity(bert_logits[i][0], entry_desc[0], dim=0))
            #     retrieved_logits.append(torch.FloatTensor(entry_descs[torch.argmax(torch.FloatTensor(sim))][0]).to(device))
            # # idx += len(entry_idx_lst)
            # batch_dict.append(torch.stack(retrieved_logits))
        
        if self.fuse == 'mean':
            # # mean
            # batch_entry_embeds = torch.cat(batch_dict)
            fused_logits = entry_ids.sum(1)
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:,0], fused_logits), dim=1))
            logits = self.classifier(self.dropout(pooled_output))
        elif self.fuse == 'attn':
            # cross attention
            # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_mask, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
            
            # fused_logits = self.dict_block(bert_logits[:,0].unsqueeze(1),
            #                             #    attention_mask=extended_attention_mask,
            #                             encoder_hidden_states=dict_logits,
            #                             encoder_attention_mask=extended_attention_mask,
            #                             )[0]
            fused_logits = self.dict_block(bert_logits,
                                        attention_mask=extended_input_mask,
                                        encoder_hidden_states=entry_ids,
                                        encoder_attention_mask=extended_attention_mask,
                                        self_attn=False,
                                        )[0]
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:, 0], fused_logits[:, 0]), dim=1))
        
            # debug
            # logits = self.bert_classifier(self.dropout(bert_pooled_logits))
            logits = self.classifier(self.dropout(pooled_output))
        elif self.fuse == 'lse':
            
            # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_ids, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
        
            fused_logits = self.dict_block(bert_logits,
                                        #    attention_mask=extended_attention_mask,
                                           encoder_hidden_states=entry_ids,
                                           encoder_attention_mask=extended_attention_mask,
                                           )[0]
            
            
            # pooled_output = self.logsumexp_pooler(torch.stack((bert_logits[:, 0], fused_logits[:, 0]), dim=1))
            # # pooled_output = self.logsumexp_pooler(torch.stack((bert_logits[:, 0], fused_logits.mean(1)), dim=1))
            
            bert_logit = self.classifier(bert_pooled_logits)
            fused_logit = self.bert_classifier(self.pooler(fused_logits))
            logits = torch.stack([bert_logit, fused_logit]) # [2, B, 15]
            logits = logits.permute(1, 0, 2)
            logits = torch.logsumexp(logits, dim=1)
        
        
            
            
            
        
        
        # # merge
        
        # bert_logits = bert_output[0]
        # # concat dict_logits
        # dict_logits = dict_output[0]
        # idx = 0
        # dict_logit_lst = []
        # for seg in seg_idx:
        #     entry_logit = dict_logits[idx:idx+seg]
        #     idx += seg
        #     # dict_logit_lst.append(entry_logit.sum(0))
        #     b, l, d = entry_logit.size()
        #     if b:
        #         dict_logit_lst.append(entry_logit.view(b*l, -1))
        #     else:
        #         dict_logit_lst.append(torch.zeros((1, self.config.hidden_size), dtype=torch.float32, device=device))
        # # padding
        # max_length = max(entry.size(0) for entry in dict_logit_lst)
        # dict_logits = torch.ones((B, max_length, self.config.hidden_size), dtype=torch.float32, device=device)
        # dict_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)
        # for ii, entry_lg in enumerate(dict_logit_lst):
        #     dict_logits[ii, :entry_lg.size(0)] = entry_lg
        #     dict_mask[ii, :entry_lg.size(0)] = torch.ones(entry_lg.size(0), dtype=torch.long)
        # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(input_dict_mask, input_dict_mask.size(), device)

        # # dict_logits = torch.stack(dict_logit_lst, dim=0)
        # # dict_logits = torch.cat(dict_logit_lst, dim=1)
        # logits = torch.cat((bert_logits, dict_logits), dim=1)
        # # pooled_output = self.mean_pooler(logits)

        # # fuse
        # fused_logits = self.dict_block(logits, attention_mask=extended_attention_mask)[0]
        # pooled_output = self.pooler(fused_logits)

        
        
        # bert_logits = bert_output[1]
        # dict_logits = dict_output[1]
        # pooled_output = self.pooler(torch.cat((bert_logits.unsqueeze(1), dict_logits.unsqueeze(1)), dim=1))
        # pooled_output = torch.cat((bert_logits, dict_logits), dim=1)
        
        

        loss = None
        if label is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        result = {'loss': loss}
        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        input_types = batch['input_types'].to(device)
        # glyph_embeds = batch['glyph_embeds'].to(device)
        # glyph_mask_ids = batch['glyph_mask_ids'].to(device)
        # entry_ids = batch['entry_ids'].to(device)
        # seg_idx = batch['entry_segs']
        input_mask = batch['input_mask'].to(device)
        B = input_ids.size(0)
        label = batch['label'].to(device)
        # clip_embeds = batch['clip_embeds'].to(device)
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=input_types,
            attention_mask=input_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # desc_ids = batch['desc_ids']
        # desc_radical_ids = batch['desc_radical_ids'].to(device)
        # desc_input_mask = batch['desc_input_mask'].to(device)
        # dict_output = self.dict_bert(
        #     input_ids=desc_ids,
        #     radical_ids=desc_radical_ids,
        #     attention_mask=desc_input_mask,
        #     # glyph_embeds=glyph_embeds,
        #     # glyph_mask_ids=glyph_mask_ids,
        #     return_dict=True,
        # )
        # label = batch['label'].to(device)
        
        # desc_idx = batch['desc_idx']
        # dict_logits = dict_output[0]
        # bert_logits = bert_output[0]
        # batch_dict = []
        # idx = 0
        # for i in range(B):
        #     entry_idx_lst = desc_idx[i]
        #     retrieved_logits = []
        #     for j in range(len(entry_idx_lst)-1):
        #         start, end = idx+entry_idx_lst[j], idx+entry_idx_lst[j+1]
        #         sim = F.cosine_similarity(bert_logits[i][0].expand(dict_logits[start: end, 0].size()), dict_logits[start:end, 0])
        #         retrieved_logits.append(dict_logits[start: end][torch.argmax(sim), 0])
        #     idx += len(entry_idx_lst)
        #     batch_dict.append(torch.stack(retrieved_logits))
        

        # batch_idx = batch['batch_idx']
        # desc_idx = batch['desc_idx']
        bert_logits = bert_output[0]
        bert_pooled_logits = bert_output[1]
        
        entry_ids = batch['entry_ids'].to(device)
        entry_mask = batch['entry_mask'].to(device)
        
        # idx = 0
        # for i in range(B):
        #     start, end = batch_idx[i], batch_idx[i+1]
        #     sim = F.cosine_similarity(bert_logits[i][0].expand(desc_ids[start:end].size()), desc_ids[start: end])
        #     retrieved_logits = []
        #     for j in range(len(desc_idx[i])-1):
        #         s, e = desc_idx[i][j], desc_idx[i][j+1]
        #         retrieved_logit = desc_ids[start:end][torch.argmax(sim[s:e])]
        #     if len(retrieved_logit.size()) != 0:
        #             retrieved_logits.append(retrieved_logit)
        #     if retrieved_logits != []:
        #         batch_dict.append(torch.stack(retrieved_logits))
        
        if self.fuse == 'mean':
            # mean
            fused_logits = entry_ids.sum(1)
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:,0], fused_logits), dim=1))
            logits = self.classifier(self.dropout(pooled_output))
            
        elif self.fuse == 'attn':
            # cross attention
            # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_mask, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
            # logits = torch.cat((bert_logits, dict_logits), dim=1)
            
            
            # fused_logits = self.dict_block(dict_logits, 
            #                                attention_mask=extended_attention_mask,
            #                                encoder_hidden_states=bert_logits[:,0].unsqueeze(1),
            #                             #    encoder_attention_mask=extended_input_mask
            #                                )[0]
            # pooled_output = self.concat_pooler(torch.cat((bert_logits[:, 0], fused_logits.mean(1)), dim=1))
            
            # fused_logits = self.dict_block(bert_logits[:,0].unsqueeze(1),
            #                             #    attention_mask=extended_attention_mask,
            #                             encoder_hidden_states=dict_logits,
            #                             encoder_attention_mask=extended_attention_mask,
            #                             )[0]
            fused_logits = self.dict_block(bert_logits,
                                        attention_mask=extended_input_mask,
                                        encoder_hidden_states=entry_ids,
                                        encoder_attention_mask=extended_attention_mask,
                                        self_attn=False,
                                        )[0]
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:, 0], fused_logits[:, 0]), dim=1))
            
            # pooled_output = self.logsumexp_pooler(torch.stack((bert_logits[:, 0], fused_logits.mean(1)), dim=1))
            # pooled_output = self.pooler(torch.cat((bert_logits[:, 0], fused_logits.sum(1)), dim=1))
            # pooled_output = torch.cat((bert_logits[:, 0], fused_logits.mean(1)), dim=1)
            # B = input_ids.size(0)
            # debug
            # logits = self.bert_classifier(self.dropout(bert_pooled_logits))
            logits = self.classifier(self.dropout(pooled_output))
        elif self.fuse == 'lse':
            
            # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_mask, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
        
            fused_logits = self.dict_block(bert_logits,
                                           attention_mask=extended_input_mask,
                                           encoder_hidden_states=entry_ids,
                                           encoder_attention_mask=extended_attention_mask,
                                           )[0]
            
            
            # pooled_output = self.logsumexp_pooler(torch.stack((bert_logits[:, 0], fused_logits[:, 0]), dim=1))
            # # pooled_output = self.logsumexp_pooler(torch.stack((bert_logits[:, 0], fused_logits.mean(1)), dim=1))
            
            bert_logit = self.classifier(bert_pooled_logits)
            fused_logit = self.bert_classifier(self.pooler(fused_logits)) #[B, 15]
            logits = torch.stack([bert_logit, fused_logit]) # [2, B, 15]
            logits = logits.permute(1, 0, 2)
            logits = torch.logsumexp(logits, dim=1)









        # bert_logits = bert_output[0]
        # # dict_logits = dict_output[0]
        # dict_logits = dict_output[0]
        # idx = 0
        # dict_logit_lst = []
        # for seg in seg_idx:
        #     entry_logit = dict_logits[idx:idx+seg]
        #     idx += seg
        #     # dict_logit_lst.append(entry_logit.sum(0))
        #     b, l, d = entry_logit.size()
        #     if b:
        #         dict_logit_lst.append(entry_logit.view(b*l, -1))
        #     else:
        #         dict_logit_lst.append(torch.zeros((1, self.config.hidden_size), dtype=torch.float32, device=device))
        # # dict_logits = torch.stack(dict_logit_lst, dim=0)
        # # logits = torch.cat((bert_logits, dict_logits), dim=1)
        # # pooled_output = self.mean_pooler(torch.cat((bert_logits, dict_logits), dim=1))
        # # padding
        # max_length = max(entry.size(0) for entry in dict_logit_lst)
        # dict_logits = torch.ones((B, max_length, self.config.hidden_size), dtype=torch.float32, device=device)
        # dict_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)
        # for ii, entry_lg in enumerate(dict_logit_lst):
        #     dict_logits[ii, :entry_lg.size(0)] = entry_lg
        #     dict_mask[ii, :entry_lg.size(0)] = torch.ones(entry_lg.size(0), dtype=torch.long)
        # input_dict_mask = torch.cat((input_mask, dict_mask), dim=1)
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(input_dict_mask, input_dict_mask.size(), device)

        # logits = torch.cat((bert_logits, dict_logits), dim=1)
        # # fuse
        # fused_logits = self.dict_block(logits, attention_mask=extended_attention_mask)[0]
        # pooled_output = self.pooler(fused_logits)

        # bert_logits = bert_output[1]
        # dict_logits = dict_output[1]
        # pooled_output = self.pooler(torch.cat((bert_logits.unsqueeze(1), dict_logits.unsqueeze(1)), dim=1))
        # # pooled_output = torch.cat((bert_logits, dict_logits), dim=1)

        

        _, pred_ans_id = logits.max(1)
        pred_ans_id = pred_ans_id.cpu().numpy()
        result = {'pred_ans': pred_ans_id}
        return result

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

class BertDictQA(BertPreTrainedModel):
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
        self.qa_output = nn.Linear(config.hidden_size*2, 2) # B, L, 2
        self.bert_qa_output = nn.Linear(config.hidden_size, 2)

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
        input_mask = batch['input_mask'].to(device)
        
        start_positions = batch['start_labels'].to(device)
        end_positions = batch['end_labels'].to(device)
        
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=input_types,
            # glyph_embeds=glyph_embeds,
            # glyph_mask_ids=glyph_mask_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        bert_logits = bert_output[0]
        bert_pooled_logits = bert_output[1]
        
        entry_ids = batch['entry_ids'].to(device)
        entry_mask = batch['entry_mask'].to(device)
        
        if self.fuse == 'sum':
            fused_logits = entry_ids.sum(1)
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:,0], fused_logits), dim=1))
            logits = self.qa_output(self.dropout(pooled_output))
        elif self.fuse == 'attn':
            # cross attention
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
            pooled_output = self.concat_pooler(torch.cat((bert_logits, fused_logits), dim=-1))
        
            # debug
            logits = self.bert_qa_output(bert_logits)
            # logits = self.qa_output(self.dropout(torch.cat((bert_logits, fused_logits), dim=-1)))
            
        elif self.fuse == 'lse':
            # logsumexp
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_mask, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
        
            fused_logits = self.dict_block(bert_logits,
                                        #    attention_mask=extended_attention_mask,
                                           encoder_hidden_states=entry_ids,
                                           encoder_attention_mask=extended_attention_mask,
                                           )[0]
            
            bert_logit = self.qa_output(bert_pooled_logits)
            fused_logit = self.bert_qa_output(self.pooler(fused_logits))
            logits = torch.stack([bert_logit, fused_logit]) # [2, B, 15]
            logits = logits.permute(1, 0, 2)
            logits = torch.logsumexp(logits, dim=1)
        
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
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
        input_mask = batch['input_mask'].to(device)
        
        start_positions = batch['start_labels'].to(device)
        end_positions = batch['end_labels'].to(device)
        
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=input_types,
            # glyph_embeds=glyph_embeds,
            # glyph_mask_ids=glyph_mask_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        bert_logits = bert_output[0]
        bert_pooled_logits = bert_output[1]
        
        entry_ids = batch['entry_ids'].to(device)
        entry_mask = batch['entry_mask'].to(device)
        
        if self.fuse == 'sum':
            fused_logits = entry_ids.sum(1)
            pooled_output = self.concat_pooler(torch.cat((bert_logits[:,0], fused_logits), dim=1))
            logits = self.qa_output(self.dropout(pooled_output))
        elif self.fuse == 'attn':
            # cross attention
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
            pooled_output = self.concat_pooler(torch.cat((bert_logits, fused_logits), dim=-1))
        
            # debug
            logits = self.bert_qa_output(bert_logits)
            # logits = self.qa_output(self.dropout(torch.cat((bert_logits, fused_logits), dim=-1)))
            
        elif self.fuse == 'lse':
            # logsumexp
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(entry_mask, entry_mask.size(), device)
            extended_input_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.size(), device)
        
            fused_logits = self.dict_block(bert_logits,
                                        #    attention_mask=extended_attention_mask,
                                           encoder_hidden_states=entry_ids,
                                           encoder_attention_mask=extended_attention_mask,
                                           )[0]
            
            bert_logit = self.qa_output(bert_pooled_logits)
            fused_logit = self.bert_qa_output(self.pooler(fused_logits))
            logits = torch.stack([bert_logit, fused_logit]) # [2, B, 15]
            logits = logits.permute(1, 0, 2)
            logits = torch.logsumexp(logits, dim=1)
        
        
        start_logits, end_logits = logits.split(1, dim=-1) # logits B, L, 2
        start_logits = start_logits.squeeze(-1).contiguous() # B, L
        end_logits = end_logits.squeeze(-1).contiguous() # B, L
        
        result = {'start_logits': start_logits, 'end_logits':end_logits}
        return result


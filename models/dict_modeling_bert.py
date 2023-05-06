
import math
from multiprocessing import pool
import random
from dataclasses import dataclass

from transformers.models.bert.modeling_bert import(
    BertAttention, BertLayer, BertEncoder, BertModel, BertForMaskedLM, BertConfig, BertForSequenceClassification, BertForPreTraining
)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput, SequenceClassifierOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.modeling_utils import apply_chunking_to_forward



logger = logging.get_logger(__name__)


class DictRetriever(BertEncoder):
    def __init__(self, config, num_layer):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([HubLayer(config) for _ in range(num_layer)])
        self.gradient_checkpointing = False
        self.hub_attn = HubAttention(config) # get weight
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        self_attn=False,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    self_attn=self_attn
                )
            hidden_states = layer_outputs[0]
        outputs = self.hub_attn(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return outputs # 0: weghted sum | 1: prob: B, num_head, query_len, value_len | 2: score
        
        
        
        
    

class HubLayer(BertLayer): 
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
        intermediate_output = self.intermediate(attention_output) # linear + act
        layer_output = self.output(intermediate_output, attention_output) # 
        return layer_output

class HubAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # self.num_attention_heads = config.num_attention_heads
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = 1
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
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
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states)) # B, num_head, e_l, head_dim
            # value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            value_layer = self.transpose_for_scores(encoder_hidden_states) # B, num_head, e_l, head_dim
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
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, q_l, head_dim * B, num_head, head_dim, e_l -> B, num_head, q_l, e_l

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
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # B, num_head, q_l, e_l

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, q_l, e_l * B, num_head, e_l, head_dim -> B, num_head, q_l, head_dim

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # -> B, q_l, head_num, head_dim
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # B, L, 
        context_layer = context_layer.view(new_context_layer_shape) # B, L, hidden_size

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = context_layer, attention_probs, attention_scores
        
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        
        if config.glyph == 'radical':
            self.radical_embeddings = nn.Embedding(config.radical_vocab_size, config.hidden_size)
            self.map_fc = nn.Linear(config.hidden_size*2, config.hidden_size)
        elif config.glyph == 'glyph':
            glyph_embedding = torch.load('datasets/glyph_embedding.pt')
            self.glyph_embeddings = nn.Embedding(config.vocab_size, embedding_dim=512, _weight=glyph_embedding)
            self.map_fc = nn.Linear(config.hidden_size+512, config.hidden_size)
        
        # self.glyph_embeddings = nn.Linear(512, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, 
        input_ids=None, 
        radical_ids=None,
        glyph_embeds=None,
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # if glyph_embeds is not None:
        #     seq_length += glyph_embeds.size()[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            # if glyph_embeds is not None:
            #     token_type_ids = torch.cat((token_type_ids, torch.ones(glyph_embeds.size()[:-1])), dim=1)

        
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        
        
        if radical_ids is not None:
            radical_embeds = self.radical_embeddings(radical_ids)
            embeds = self.map_fc(torch.cat((inputs_embeds, radical_embeds), dim=2))
        if glyph_embeds is not None:
            glyph_embeds = self.glyph_embeddings(input_ids)
            embeds = self.map_fc(torch.cat((inputs_embeds, glyph_embeds), dim=2))
        else:
            embeds = inputs_embeds

        token_type_embeddings = self.token_type_embeddings(token_type_ids)    
        embeddings = embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertClipModel(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # self.init_weights()
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        radical_ids=None,
        glyph_embeds=None,
        glyph_mask_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            # if glyph_embeds is not None:
            #     token_type_ids = torch.cat((token_type_ids, torch.ones(glyph_embeds.size()[:-1], dtype=torch.long, device=device)), dim=1)
        # if glyph_mask_ids is not None:
        #     attention_mask = torch.cat((attention_mask, glyph_mask_ids), dim=1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            radical_ids=radical_ids,
            glyph_embeds=glyph_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

@dataclass
class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None





class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertForDictPretraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

        self.dict_retriever = DictRetriever(config, 1)
        self.oribert = BertModel(config)
        self.bert = BertClipModel(config)
        self.cls = BertPreTrainingHeads(config)

        # self.init_weights()
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        radical_ids=None,
        glyph_embeds=None,
        glyph_mask_ids=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        entry_input_ids=None,
        entry_radical_ids=None,
        entry_input_mask=None,
        entry_glyph_embeds=None,
        syno_input_ids=None,
        syno_radical_ids=None,
        syno_input_mask=None,
        anto_input_ids=None,
        anto_radical_ids=None,
        anto_input_mask=None,
        syno_glyph_embeds=None,
        syno_glyph_mask_ids=None,
        anto_glyph_embeds=None,
        anto_glyph_mask_ids=None,
        example_input_ids=None,
        example_mask_ids=None,
        mean_input_ids=None,
        flatten_mean_input_ids=None,
        mean_mask_ids=None,
        flatten_mean_mask_ids=None,
        mean_label_ids=None,
        flatten_mean_radical_ids=None,
        mean_glyph_embeds=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            outputs = self.bert(
                input_ids,
                radical_ids=radical_ids,
                attention_mask=attention_mask,
                glyph_embeds=glyph_embeds,
                glyph_mask_ids=glyph_mask_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)       

        if entry_input_ids is not None:
            entry_outputs = self.bert(
                entry_input_ids,
                attention_mask=entry_input_mask,
                radical_ids=entry_radical_ids,
                glyph_embeds=entry_glyph_embeds,
                return_dict=return_dict
            )
            entry_sequence_output = entry_outputs[0]
        
        if anto_input_ids is not None:
            anto_outputs = self.bert(
                anto_input_ids,
                attention_mask=anto_input_mask,
                radical_ids=anto_radical_ids,
                glyph_embeds=anto_glyph_embeds,
                glyph_mask_ids=anto_glyph_mask_ids,
                return_dict=return_dict,
            )
            anto_sequence_output = anto_outputs[0]

        if syno_input_ids is not None:
            syno_outputs = self.bert(
                syno_input_ids,
                attention_mask=syno_input_mask,
                radical_ids=syno_radical_ids,
                glyph_embeds=syno_glyph_embeds,
                glyph_mask_ids=syno_glyph_mask_ids,
                return_dict=return_dict,
            )
            syno_sequence_output = syno_outputs[0]
            
        if example_input_ids is not None:
            with torch.no_grad():
                example_outputs = self.oribert(
                    example_input_ids,
                    attention_mask=example_mask_ids
                )
            example_logits = example_outputs[0] # B, seq_len, hidden_size
            flatten_mean_outputs = self.bert(
                flatten_mean_input_ids,
                attention_mask=flatten_mean_mask_ids,
                radical_ids=flatten_mean_radical_ids,
                glyph_embeds=mean_glyph_embeds,
                return_dict=return_dict,
            ) # -> B*num_choices, num_seq, hidden_size
            mean_logits = flatten_mean_outputs[0][:, 0].contiguous().view(-1, mean_input_ids.size(1), self.config.hidden_size) # B, max_means, hidden_size
            extend_mean_mask_attention: torch.Tensor = self.get_extended_attention_mask(mean_mask_ids, mean_mask_ids.size())
            
            retrieval_outputs = self.dict_retriever(
                example_logits[:, 0].unsqueeze(1), # B, 
                encoder_hidden_states=mean_logits,
                encoder_attention_mask=extend_mean_mask_attention,
                self_attn=False,
            )
            retrieval_score = retrieval_outputs[2].mean(1).squeeze(1) # B, num_head, q_l, e_l -> B, e_l
            retrieval_prob = retrieval_outputs[1].mean(1).squeeze(1) # B,e_l
            
            
            
            

        # mask character loss

        # contrastive loss

          

        total_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            # total_loss = masked_lm_loss + next_sentence_loss
            total_loss = masked_lm_loss

        if labels is not None and syno_input_ids is not None:
            cl_loss = self.cl_loss(entry_sequence_output[:, 0], syno_sequence_output[:, 0], anto_sequence_output[:, 0])
            # total_loss = 0.4*masked_lm_loss + 0.6*cl_loss
        # if labels is not None and next_sentence_label is not None:
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        #     # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        #     # total_loss = masked_lm_loss + next_sentence_loss
        #     total_loss = masked_lm_loss
        if mean_label_ids is not None:
            if len(mean_label_ids.size()) > 1:
                mean_label_ids = mean_label_ids.squeeze(-1)
            # generate prob (B, 1) -> (B, e_l)
            ignored_index = retrieval_score.size(1)
            mean_label_ids = mean_label_ids.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
            eg_loss = loss_fct(retrieval_score, mean_label_ids) # B, e_l |  B, 1
            
            # print(retrieval_prob)
            # print(mean_label_ids)
            # mean_label_prob = F.one_hot(mean_label_ids, num_classes=retrieval_prob.size(1)).to(retrieval_prob.dtype)
            # loss_fct = MSELoss(reduction='none')
            # eg_loss = loss_fct(retrieval_prob, mean_label_prob).mean(1)
            
            
        loss_tp = tuple()
        if input_ids is not None:
            loss_tp = loss_tp + (masked_lm_loss, )
        else:
            loss_tp = loss_tp + (None, )
        if syno_input_ids is not None:
            loss_tp = loss_tp + (cl_loss, )
        else:
            loss_tp = loss_tp + (None, )
        if example_input_ids is not None:
            loss_tp = loss_tp + (eg_loss, )
        else:
            loss_tp = loss_tp + (None, )
            
        
        
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (loss_tp + output) if total_loss is not None else output
        if len(loss_tp) == 1:
            loss_tp = loss_tp[0]
        if not self.eg_only:
            return BertForPreTrainingOutput(
                    loss=loss_tp,
                    prediction_logits=prediction_scores,
                    seq_relationship_logits=seq_relationship_score,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            return BertForPreTrainingOutput(
                loss=loss_tp,
            )
    def cl_loss(self, ori_hidden_states, syno_hidden_states, anto_hidden_states):
        # input [B, hidden_size]
        # -> [B]
        syn_hs = torch.einsum('ij,ij->i', [ori_hidden_states, syno_hidden_states])
        ant_hs = torch.einsum('ij,ij->i', [ori_hidden_states, anto_hidden_states])
        loss_func = nn.LogSoftmax(dim=0)
        hs = torch.stack([syn_hs, ant_hs], dim=0)
        # print(syn_hs)
        # print(ant_hs)
        ls = loss_func(hs)[0]
        return -ls


class BertForMaskedLM(BertForMaskedLM):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertClipModel(config, add_pooling_layer=False)
        # self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        reduce_loss=False,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertForDictSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertClipModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        glyph_embeds=None,
        glyph_mask_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            glyph_embeds=glyph_embeds,
            glyph_mask_ids=glyph_mask_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


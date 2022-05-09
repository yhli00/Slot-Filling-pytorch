import torch
import torch.nn as nn
# from transformers import BertModel
import sys
from transformers import BertModel
import torch.nn.functional as F
import math


class BertMRC(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.start_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.start_output_layer.bias.data.zero_()
        self.end_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.end_output_layer.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]
        start_logits = self.start_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
        end_logits = self.end_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
        return start_logits, end_logits




# class BertLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(BertLayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias


# class BertSelfAttention(nn.Module):
#     def __init__(self, config):
#         super(BertSelfAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.parse_query = nn.Linear(config.hidden_size, self.attention_head_size)
#         self.parse_key = nn.Linear(config.hidden_size, self.attention_head_size)
#         self.parse_value = nn.Linear(config.hidden_size, self.attention_head_size)

#         self.mlp = nn.Linear(self.all_head_size + self.attention_head_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def transpose_for_scores_parse(self, x):
#         new_x_shape = x.size()[:-1] + (1, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states, attention_mask, span_mask=None):
#         # bsz, seq_len, dim
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         # bsz, seq_len, head_size
#         # mixed_parse_query_layer = self.parse_query(hidden_states)
#         # mixed_parse_key_layer = self.parse_key(hidden_states)
#         mixed_parse_value_layer = self.parse_value(hidden_states)

#         # bsz, 1, seq_len, head_size
#         # parse_query_layer = self.transpose_for_scores_parse(mixed_parse_query_layer)
#         # parse_key_layer = self.transpose_for_scores_parse(mixed_parse_key_layer)
#         parse_value_layer = self.transpose_for_scores_parse(mixed_parse_value_layer)
#         if span_mask is not None:
#             parse_context_layer = torch.matmul(span_mask, parse_value_layer)
#             parse_context_layer = parse_context_layer.permute(0, 2, 1, 3).contiguous()
#         # bsz, 1, seq_len, seq_len
#         # parse_score = torch.matmul(parse_query_layer, parse_key_layer.transpose(-1,-2))

#         # bsz, num_head, seq_len, head_size
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # bsz, num_head, seq_len, seq_len
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # if span_mask is not None:
#         #     print("att_score", attention_scores.shape)
#         #     print("att_mask", attention_mask.shape)

#         attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         if span_mask is not None:
#             context_layer = torch.cat([context_layer, parse_context_layer], dim=-2)
#             new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size + self.attention_head_size,)
#             # bsz, seq_len, (num_head+1)*head_size
#             context_layer = context_layer.view(*new_context_layer_shape)
#             context_layer = self.mlp(context_layer)
#         else:
#             new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#             context_layer = context_layer.view(*new_context_layer_shape)
#         return context_layer


# class BertSelfOutput(nn.Module):
#     def __init__(self, config):
#         super(BertSelfOutput, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states


# class BertAttention(nn.Module):
#     def __init__(self, config):
#         super(BertAttention, self).__init__()
#         self.self = BertSelfAttention(config)
#         self.output = BertSelfOutput(config)

#     def forward(self, input_tensor, attention_mask, span_mask=None):
#         self_output = self.self(input_tensor, attention_mask, span_mask)
#         attention_output = self.output(self_output, input_tensor)
#         return attention_output

# def gelu(x):
#     """Implementation of the gelu activation function.
#         For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
#         0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#         Also see https://arxiv.org/abs/1606.08415
#     """
#     return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# def swish(x):
#     return x * torch.sigmoid(x)


# ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

# class BertIntermediate(nn.Module):
#     def __init__(self, config):
#         super(BertIntermediate, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states


# class BertOutput(nn.Module):
#     def __init__(self, config):
#         super(BertOutput, self).__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states


# class BertLayer(nn.Module):
#     def __init__(self, config):
#         super(BertLayer, self).__init__()
#         self.attention = BertAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self, hidden_states, attention_mask, span_mask=None):
#         attention_output = self.attention(hidden_states, attention_mask, span_mask)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output

# class BertMRC(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(pretrained_model)
#         self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.attention_layer = BertLayer(self.bert.config)
#         self.dropout = nn.Dropout(p=0.1)
#         self.start_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.start_output_layer.bias.data.zero_()
#         self.end_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.end_output_layer.bias.data.zero_()
    
#     def forward(self, context_input_ids, context_attention_mask, context_token_type_ids,
#                 query_input_ids, query_attention_mask, query_token_type_ids):
        

#         context_output = self.bert(
#             input_ids=context_input_ids,
#             attention_mask=context_attention_mask,
#             token_type_ids=context_token_type_ids
#         )
#         encoded_context = context_output.last_hidden_state  # [B, L1, H]
#         query_output = self.bert(
#             input_ids=query_input_ids,
#             attention_mask=query_attention_mask,
#             token_type_ids=query_token_type_ids
#         )
#         encoded_query = query_output.last_hidden_state  # [B, L2, H]

#         encoded_context = self.dropout(encoded_context)
#         encoded_query = self.dropout(encoded_query)

#         B, L1, H = encoded_context.shape
#         _, L2, _ = encoded_query.shape
        
#         encoded_query_context = torch.cat([encoded_query, encoded_context], dim=1)  # [B, L1+L2, H]
#         query_context_attention_mask = torch.cat([query_attention_mask, context_attention_mask], dim=1)  # [B, L1+L2]
#         query_context_attention_mask = query_context_attention_mask.unsqueeze(-2).repeat(1, L1+L2, 1)  # [B, L1+L2, L1+L2]
#         query_context_attention_mask = query_context_attention_mask.unsqueeze(1)  # [B, 1, L1+L2, L1+L2]
#         query_context_attention_mask = (1.0 - query_context_attention_mask) * (-100000.0)

#         query_context_output = self.attention_layer(encoded_query_context, query_context_attention_mask)  # [B, L1+L2, H]

#         last_hidden_state = query_context_output[:, L2:, :]  # [B, L1, H]

#         start_logits = self.start_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         end_logits = self.end_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         return start_logits, end_logits

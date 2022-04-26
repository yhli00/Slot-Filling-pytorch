import torch
from torch import nn
from transformers import BertModel, RobertaModel
import math
import sys
import torch.nn.functional as F

# class BERTPretrainedMRC(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = BertForQuestionAnswering.from_pretrained(pretrained_model)

    
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         """
#         Args:
#             input_ids: bert input tokens, tensor of shape [seq_len]
#             token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
#             attention_mask: attention mask, tensor of shape [seq_len]
#         Returns:
#             start_logits: start/non-start probs of shape [seq_len]
#             end_logits: end/non-end probs of shape [seq_len]
#             match_logits: start-end-match probs of shape [seq_len, 1]
#         """
#         bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         start_logits = bert_outputs.start_logits  # [B, seq_len]
#         end_logits = bert_outputs.end_logits
#         return start_logits, end_logits


# class BertMRC(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(pretrained_model)
#         self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.start_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.start_output_layer.bias.data.zero_()
#         self.end_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.end_output_layer.bias.data.zero_()
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         bert_outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]
#         start_logits = self.start_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         end_logits = self.end_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         return start_logits, end_logits


# class BertMRC(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(pretrained_model)
#         # self.bert2 = BertModel.from_pretrained(pretrained_model)
#         self.dropout = nn.Dropout(p=0.05)
#         self.fc_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         self.fc_3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
#         self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.start_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.start_output_layer.bias.data.zero_()
#         self.end_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.end_output_layer.bias.data.zero_()
    
#     def forward(self, input_ids, attention_mask, token_type_ids,
#                 label_knowledge_input_ids, label_knowledge_attention_mask, label_knowledge_token_type_ids):
#         '''
#         input_ids: [B, L]
#         label_knowledge_input_ids: [B, L]
#         '''
#         # bert_outputs = self.bert(
#         #     input_ids=input_ids,
#         #     attention_mask=attention_mask,
#         #     token_type_ids=token_type_ids
#         # )
#         # last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]
#         L = input_ids.shape[1]
#         context_output = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         context_encode = context_output.last_hidden_state  # [B, L, H]
#         label_knowledge_output = self.bert(
#             input_ids=label_knowledge_input_ids,
#             attention_mask=label_knowledge_attention_mask,
#             token_type_ids=label_knowledge_token_type_ids
#         )
#         label_knowledge_encode = label_knowledge_output.last_hidden_state  # [B, L, H]
        
#         context_encode = self.dropout(context_encode)
#         label_knowledge_encode = self.dropout(label_knowledge_encode)

#         # context_feature = self.fc_1(context_encode)  # [B, L, H]
#         # label_feature = self.fc_2(label_knowledge_encode)  # [B, L, H]
#         context_feature = context_encode
#         label_feature = label_knowledge_encode

#         label_feature_t = label_feature.transpose(1, 2)  # [B, H, L]
#         score = torch.bmm(context_feature, label_feature_t)  # [B, L, L] 第一维的L对应context
#         label_attention_mask_extand = label_knowledge_attention_mask.unsqueeze(1).repeat(1, L, 1)  # [B, L]->[B, 1, L]->[B, L, L]
#         score = score + (1 - label_attention_mask_extand) * (-999999.0)
#         score = F.softmax(score, dim=-1)  # [B, L, L]
#         weighted_label_feature = torch.bmm(score, label_feature)  # [B, L, L] * [B, L, H]->[B, L, H]
#         fused_feature = context_feature + weighted_label_feature
#         # fused_feature = torch.tanh(self.fc_3(fused_feature))  # [B, L, H]


#         start_logits = self.start_output_layer(fused_feature).squeeze(-1)  # [B, L]
#         end_logits = self.end_output_layer(fused_feature).squeeze(-1)  # [B, L]
#         return start_logits, end_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.parse_query = nn.Linear(config.hidden_size, self.attention_head_size)
        self.parse_key = nn.Linear(config.hidden_size, self.attention_head_size)
        self.parse_value = nn.Linear(config.hidden_size, self.attention_head_size)

        self.mlp = nn.Linear(self.all_head_size + self.attention_head_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_parse(self, x):
        new_x_shape = x.size()[:-1] + (1, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, span_mask=None):
        # bsz, seq_len, dim
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # bsz, seq_len, head_size
        # mixed_parse_query_layer = self.parse_query(hidden_states)
        # mixed_parse_key_layer = self.parse_key(hidden_states)
        mixed_parse_value_layer = self.parse_value(hidden_states)

        # bsz, 1, seq_len, head_size
        # parse_query_layer = self.transpose_for_scores_parse(mixed_parse_query_layer)
        # parse_key_layer = self.transpose_for_scores_parse(mixed_parse_key_layer)
        parse_value_layer = self.transpose_for_scores_parse(mixed_parse_value_layer)
        if span_mask is not None:
            parse_context_layer = torch.matmul(span_mask, parse_value_layer)
            parse_context_layer = parse_context_layer.permute(0, 2, 1, 3).contiguous()
        # bsz, 1, seq_len, seq_len
        # parse_score = torch.matmul(parse_query_layer, parse_key_layer.transpose(-1,-2))

        # bsz, num_head, seq_len, head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # bsz, num_head, seq_len, seq_len
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # if span_mask is not None:
        #     print("att_score", attention_scores.shape)
        #     print("att_mask", attention_mask.shape)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        if span_mask is not None:
            context_layer = torch.cat([context_layer, parse_context_layer], dim=-2)
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size + self.attention_head_size,)
            # bsz, seq_len, (num_head+1)*head_size
            context_layer = context_layer.view(*new_context_layer_shape)
            context_layer = self.mlp(context_layer)
        else:
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, span_mask=None):
        self_output = self.self(input_tensor, attention_mask, span_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, span_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, span_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output





class BertMRC(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = self.bert.config
        self.span_layer = BertLayer(self.config)
        self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.gamma = nn.Parameter(torch.ones(1))
        self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        
        self.span_layer.apply(self.init_bert_weights)
        self.qa_outputs.apply(self.init_bert_weights)
        self.start_output_layer.apply(self.init_bert_weights)
        self.end_output_layer.apply(self.init_bert_weights)
    
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, input_span_mask=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]

        extended_span_attention_mask = input_span_mask.unsqueeze(1)  # [B, 1, L, L]
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        span_sequence_output = self.span_layer(last_hidden_state, extended_span_attention_mask)
        
        w = F.softmax(self.w, dim=0)
        sequence_output = self.gamma * (w[0] * last_hidden_state + w[1] * span_sequence_output)  # [B, L, H]

        sequence_output = last_hidden_state
        logits = self.qa_outputs(sequence_output)  # [B, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)  # [B, L, 1]
        start_logits = start_logits.squeeze(-1)  # [B, L]
        end_logits = end_logits.squeeze(-1)  # [B, L]



        start_logits = self.start_output_layer(sequence_output).squeeze(-1)  # [B, L]
        end_logits = self.end_output_layer(sequence_output).squeeze(-1)  # [B, L]
        return start_logits, end_logits

# class BertForQuestionAnsweringSpanMask(BertPreTrainedModel):
#     """BERT model for Question Answering (span extraction).
#     This module is composed of the BERT model with a linear layer on top of
#     the sequence output that computes start_logits and end_logits
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])
#     config = BertConfig(vocab_size=32000, hidden_size=512,
#         num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
#     model = BertForQuestionAnswering(config)
#     start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """

#     def __init__(self, config):
#         super(BertForQuestionAnsweringSpanMask, self).__init__(config)
#         self.bert = BertModel(config)
#         self.span_layer = BertLayer(config)
#         self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))

#         self.gamma = nn.Parameter(torch.ones(1))
#         self.dropout = nn.Dropout(0.3)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_span_mask=None,
#                 start_positions=None, end_positions=None, is_impossibles=None):
#         bert_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
#                                    output_all_encoded_layers=False)

#         # span_attention_mask
#         extended_span_attention_mask = input_span_mask.unsqueeze(1)
#         extended_span_attention_mask = extended_span_attention_mask.to(
#             dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0

#         span_sequence_output = self.span_layer(bert_output, extended_span_attention_mask)

#         w = F.softmax(self.w)
#         sequence_output = self.gamma * (w[0] * bert_output + w[1] * span_sequence_output)

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             if len(is_impossibles.size()) > 1:
#                 is_impossibles = is_impossibles.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#             is_impossibles.clamp_(0, ignored_index)

#             # loss CE
#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = start_loss + end_loss

#             return total_loss
#         else:
#             return start_logits, end_logits


# class BertMRC(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = RobertaModel.from_pretrained(pretrained_model)
#         self.start_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.end_output_layer = nn.Linear(self.bert.config.hidden_size, 1)
#         self.start_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.start_output_layer.bias.data.zero_()
#         self.end_output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.end_output_layer.bias.data.zero_()
    
#     def forward(self, input_ids, attention_mask=None):
#         bert_outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]
#         start_logits = self.start_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         end_logits = self.end_output_layer(last_hidden_state).squeeze(-1)  # [B, L]
#         return start_logits, end_logits

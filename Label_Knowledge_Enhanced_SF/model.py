from unicodedata import bidirectional
import torch
import torch.nn as nn
# from transformers import BertModel
import sys
from transformers import DebertaV2Model, BertModel
from modeling_bart import BartModel
import torch.nn.functional as F
import math





class LabelEnhancedBartMrc(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bart = BartModel.from_pretrained(pretrained_model)
        self.config = self.bart.config
        self.entity_start_classifier = nn.Linear(self.bart.config.d_model, 1)
        self.entity_end_classifier = nn.Linear(self.bart.config.d_model, 1)
        self.entity_start_classifier.apply(self._init_weights)
        self.entity_end_classifier.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, text_input_ids, text_attention_mask, label_input_ids, label_attention_mask):
        '''
        text_input_ids: [B, L1]
        text_attention_mask: [B, L1]
        label_input_ids: [num_labels, L2]
        label_attention_mask: [num_labels, L2]
        '''
        B, L1 = text_input_ids.shape
        num_labels, L2 = label_input_ids.shape
        encoder_output = self.bart.encoder(
            input_ids=label_input_ids,
            attention_mask=label_attention_mask
        )
        encoder_output = encoder_output.last_hidden_state  # [num_labels, L2, H]
        _, _, H = encoder_output.shape
        encoder_output = encoder_output.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, num_labels, L2, H]
        encoder_output = encoder_output.reshape(B * num_labels, L2, H)  # [B*num_labels, L2, H]
        encoder_attention_mask = label_attention_mask.unsqueeze(0).repeat(B, 1, 1)  # [B, num_labels, L2]
        encoder_attention_mask = encoder_attention_mask.reshape(B * num_labels, L2)

        decoder_input_ids = text_input_ids  # [B, L1]
        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_labels, 1)  # [B, num_labels, L1]
        decoder_input_ids = decoder_input_ids.reshape(B * num_labels, L1)
        decoder_attention_mask = text_attention_mask  # [B, L1]
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_labels, 1)  # [B, num_labels, L1]
        decoder_attention_mask = decoder_attention_mask.reshape(B * num_labels, L1)

        decoder_output = self.bart.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask
        )
        decoder_output = decoder_output.last_hidden_state  # [B*num_labels, L1, H]
        
        start_logits = self.entity_start_classifier(decoder_output)
        end_logits = self.entity_end_classifier(decoder_output)

        start_logits = start_logits.squeeze(-1)  # [B*num_labels, L1]
        end_logits = end_logits.squeeze(-1)  # [B*num_labels, L1]

        start_logits = start_logits.reshape(B, num_labels, L1)
        end_logits = end_logits.reshape(B, num_labels, L1)
        start_logits = start_logits.transpose(1, 2)
        end_logits = end_logits.transpose(1, 2)

        return start_logits, end_logits  # [B, L1, num_labels]





















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


class LabelEnhanceBert(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        if pretrained_model == 'microsoft/deberta-v3-large':
            self.bert = DebertaV2Model.from_pretrained(pretrained_model)
        elif pretrained_model == 'bert-large-uncased':
            self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = self.bert.config
        # self.pos_embedding = nn.Embedding(19, 512)
        # self.ent_embedding = nn.Embedding(19, 512)
        self.dropout = nn.Dropout(p=0.2)
        # self.fc_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
        # self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
        # self.fc_3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.entity_start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.entity_end_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.attention_layer = BertLayer(self.config)
        
        # self._init_weights(self.pos_embedding)
        # self._init_weights(self.ent_embedding)
        # self.fc_1.apply(self._init_weights)
        # self.fc_2.apply(self._init_weights)
        # self.fc_3.apply(self._init_weights)
        self.entity_start_classifier.apply(self._init_weights)
        self.entity_end_classifier.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, text_input_ids, text_attention_mask, text_token_type_ids, label_input_ids, 
                label_attention_mask, label_token_type_ids, text_pos_token=None, text_ent_token=None,
                label_pos_token=None, label_ent_token=None):
        '''
        text_input_ids: [B, L1]
        text_attention_mask: [B, L1]
        text_token_type_ids: [B, L1]
        label_input_ids: [num_label, L2]
        label_attention_mask: [num_label, L2]
        label_token_type_ids: [num_label, L2]
        text_pos_token: [B, L1]
        text_ent_token: [B, L1]
        label_pos_token: [num_label, L2]
        label_ent_token: [num_label, L2]
        '''
        text_output = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids
        )
        encode_text = text_output[0]  # [B, L1, H]
        label_output = self.bert(
            input_ids=label_input_ids,
            attention_mask=label_attention_mask,
            token_type_ids=label_token_type_ids
        )
        encode_label = label_output[0]  # [num_labels, L2, H]
        encode_text = self.dropout(encode_text)
        encode_label = self.dropout(encode_label)
        B, L1, H = encode_text.shape
        num_labels, L2, _ = encode_label.shape
        encode_text = encode_text.unsqueeze(1).repeat(1, num_labels, 1, 1)  # [B, L1, H]->[B, 1, L1, H]->[B, num_labels, L1, H]
        encode_label = encode_label.unsqueeze(0).repeat(B, 1, 1, 1)  # [num_labels, L2, H]->[1, num_labels, L2, H]->[B, num_labels, L2, H]
        label_attn_mask_extend = label_attention_mask.unsqueeze(0).repeat(B, 1, 1)  # [num_labels, L2]->[1, num_labels, L2]->[B, num_labels, L2]
        text_attn_mask_extend = text_attention_mask.unsqueeze(1).repeat(1, num_labels, 1)  # [B, L1]->[B, 1, L1]->[B, num_labels, L1]
        attention_mask_extend = torch.cat([label_attn_mask_extend, text_attn_mask_extend], dim=-1)  # [B, num_labels, L1+L2]
        attention_mask_extend = attention_mask_extend.unsqueeze(-1).repeat(1, 1, 1, L1+L2)  # [B, num_labels, L1+L2]->[B, num_labels, L1+L2, L1+L2]
        attention_mask_extend = attention_mask_extend.reshape(B * num_labels, L1+L2, L1+L2)  # [B*num_labels, L1+L2, L1+L2]
        attention_mask_extend = attention_mask_extend.unsqueeze(1)
        attention_mask_extend = (1.0 - attention_mask_extend) * (-100000.0)
        hidden_state_extend = torch.cat([encode_label, encode_text], dim=-2)  # [B, num_labels, L1+L2, H]
        hidden_state_extend = hidden_state_extend.reshape(B * num_labels, L1+L2, H)  # [B*num_labels, L1+L2, H]
        seq_output = self.attention_layer(hidden_state_extend, attention_mask_extend)  # [B*num_labels, L1+L2, H]
        seq_output = seq_output.reshape(B, num_labels, L1+L2, H)
        context_output = seq_output[:, :, L2:, :]  # [B, num_labels, L2, H]
        assert context_output.shape[2] == L1
        start_logits = self.entity_start_classifier(context_output)  # [B, num_labels, L1, 1]
        end_logits = self.entity_end_classifier(context_output)  # [B, num_labels, L1, 1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_logits = start_logits.transpose(1, 2)
        end_logits = end_logits.transpose(1, 2)
        return start_logits, end_logits  # [B, L1, num_labels]



# class LabelEnhanceBert(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         pretrained_model = args.pretrained_model
#         if pretrained_model == 'microsoft/deberta-v3-large':
#             self.bert = DebertaV2Model.from_pretrained(pretrained_model)
#         else:
#             self.bert = BertModel.from_pretrained(pretrained_model)
#         self.config = self.bert.config
#         # self.pos_embedding = nn.Embedding(19, 512)
#         # self.ent_embedding = nn.Embedding(19, 512)
#         self.dropout = nn.Dropout(p=args.dropout_rate)
#         # self.fc_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         # self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         # self.fc_3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
#         self.entity_start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
#         self.entity_end_classifier = nn.Linear(self.bert.config.hidden_size, 1)

#         # self.attention_layers = nn.ModuleList([BertLayer(self.config) for _ in range(1)])
#         self.attention_layer = BertLayer(self.config)
#         # self.attention_layer = BertLayer(self.config)
#         # self.attention_layer2 = BertLayer(self.config)
#         # self.attention_layer3 = BertLayer(self.config)
        
#         # self._init_weights(self.pos_embedding)
#         # self._init_weights(self.ent_embedding)
#         # self.fc_1.apply(self._init_weights)
#         # self.fc_2.apply(self._init_weights)
#         # self.fc_3.apply(self._init_weights)
#         self.entity_start_classifier.apply(self._init_weights)
#         self.entity_end_classifier.apply(self._init_weights)
#         # for attention_layer in self.attention_layers:
#         # self.attention_layer.apply(self._init_weights)
#         # self.attention_layer.apply(self._init_weights)
#         # self.attention_layer2.apply(self._init_weights)
#         # self.attention_layer3.apply(self._init_weights)

#         # self.att_weight_c = nn.Linear(self.config.hidden_size, 1)
#         # self.att_weight_q = nn.Linear(self.config.hidden_size, 1)
#         # self.att_weight_cq = nn.Linear(self.config.hidden_size, 1)
#         # self.att_weight_c.apply(self._init_weights)
#         # self.att_weight_q.apply(self._init_weights)
#         # self.att_weight_cq.apply(self._init_weights)


#     def attention_flow_layer(self, c, q):
#         """
#         :param c: (B, c_len, H)
#         :param q: (batch, q_len, H)
#         :return: (batch, c_len, q_len)
#         """
#         c_len = c.size(1)
#         q_len = q.size(1)

#         # (batch, c_len, q_len, hidden_size * 2)
#         #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
#         # (batch, c_len, q_len, hidden_size * 2)
#         #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
#         # (batch, c_len, q_len, hidden_size * 2)
#         #cq_tiled = c_tiled * q_tiled
#         #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

#         cq = []
#         for i in range(q_len):
#             #(batch, 1, hidden_size * 2)
#             qi = q.select(1, i).unsqueeze(1)
#             #(batch, c_len, 1)
#             ci = self.att_weight_cq(c * qi).squeeze()
#             cq.append(ci)
#         # (batch, c_len, q_len)
#         cq = torch.stack(cq, dim=-1)

#         # (batch, c_len, q_len)
#         s = self.att_weight_c(c).expand(-1, -1, q_len) + \
#             self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
#             cq

#         # (batch, c_len, q_len)
#         a = F.softmax(s, dim=2)
#         # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
#         c2q_att = torch.bmm(a, q)
#         # (batch, 1, c_len)
#         b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
#         # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
#         q2c_att = torch.bmm(b, c).squeeze()
#         # (batch, c_len, hidden_size * 2) (tiled)
#         q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
#         # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

#         # (batch, c_len, H * 4)
#         x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
#         return x


#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


#     def forward(self, text_input_ids, text_attention_mask, text_token_type_ids, label_input_ids, 
#                 label_attention_mask, label_token_type_ids, text_pos_token=None, text_ent_token=None,
#                 label_pos_token=None, label_ent_token=None):
#         '''
#         text_input_ids: [B, L1]
#         text_attention_mask: [B, L1]
#         text_token_type_ids: [B, L1]
#         label_input_ids: [num_label, L2]
#         label_attention_mask: [num_label, L2]
#         label_token_type_ids: [num_label, L2]
#         text_pos_token: [B, L1]
#         text_ent_token: [B, L1]
#         label_pos_token: [num_label, L2]
#         label_ent_token: [num_label, L2]
#         '''
#         text_output = self.bert(
#             input_ids=text_input_ids,
#             attention_mask=text_attention_mask,
#             token_type_ids=text_token_type_ids,
#             # output_hidden_states=True
#         )
#         encode_text = text_output[0]  # [B, L1, H]
#         label_output = self.bert(
#             input_ids=label_input_ids,
#             attention_mask=label_attention_mask,
#             token_type_ids=label_token_type_ids,
#             # output_hidden_states=True
#         )
#         encode_label = label_output[0]  # [num_labels, L2, H]
        
#         # label_output = torch.stack(label_output[1:], dim=0)
#         # text_output = torch.stack(text_output[1:], dim=0)


#         # encode_text = torch.mean(text_output, dim=0)  # [B, L1, H]
#         # encode_label = torch.mean(label_output, dim=0)  # [B, L2, H]


#         encode_text = self.dropout(encode_text)
#         encode_label = self.dropout(encode_label)


#         B, L1, H = encode_text.shape
#         num_labels, L2, _ = encode_label.shape

#         encode_text = encode_text.unsqueeze(1).repeat(1, num_labels, 1, 1)  # [B, L1, H]->[B, 1, L1, H]->[B, num_labels, L1, H]
#         encode_label = encode_label.unsqueeze(0).repeat(B, 1, 1, 1)  # [num_labels, L2, H]->[1, num_labels, L2, H]->[B, num_labels, L2, H]

#         # context_output = self.attention_flow_layer(encode_text, encode_label)  # [B*num_labels, L1, H*4]
#         # context_output = context_output.reshape(B, num_labels, L1, H * 4)


#         label_attn_mask_extend = label_attention_mask.unsqueeze(0).repeat(B, 1, 1)  # [num_labels, L2]->[1, num_labels, L2]->[B, num_labels, L2]
#         text_attn_mask_extend = text_attention_mask.unsqueeze(1).repeat(1, num_labels, 1)  # [B, L1]->[B, 1, L1]->[B, num_labels, L1]

#         attention_mask_extend_tmp = torch.cat([label_attn_mask_extend, text_attn_mask_extend], dim=-1)  # [B, num_labels, L1+L2]
#         # attention_mask_extend = attention_mask_extend_tmp[:, :, None, :]  # [B, num_labels, 1, L1+L2]
#         # attention_mask_extend = attention_mask_extend_tmp.unsqueeze(-2).repeat(1, 1, L1+L2, 1)  # [B, num_labels, L1+L2]->[B, num_labels, L1+L2, L1+L2]
#         attention_mask_extend = attention_mask_extend_tmp.unsqueeze(-1).repeat(1, 1, 1, L1+L2)  # [B, num_labels, L1+L2]->[B, num_labels, L1+L2, L1+L2]
#         # attention_mask_extend = attention_mask_extend_1 + attention_mask_extend_2
#         # attention_mask_extend = attention_mask_extend == 2
#         # attention_mask_extend = attention_mask_extend.long()
#         attention_mask_extend = attention_mask_extend.reshape(B * num_labels, L1+L2, L1+L2)  # [B*num_labels, L1+L2, L1+L2]
#         attention_mask_extend = attention_mask_extend.unsqueeze(1)
#         attention_mask_extend = (1.0 - attention_mask_extend) * (-100000.0)

#         hidden_state_extend = torch.cat([encode_label, encode_text], dim=-2)  # [B, num_labels, L1+L2, H]
#         hidden_state_extend = hidden_state_extend.reshape(B * num_labels, L1+L2, H)  # [B*num_labels, L1+L2, H]

#         hidden_state = hidden_state_extend
#         # for attention_layer in self.attention_layers:
#         hidden_state = self.attention_layer(hidden_state, attention_mask_extend)
#         seq_output = hidden_state.reshape(B, num_labels, L1+L2, H)
#         context_output = seq_output[:, :, L2:, :]  # [B, num_labels, L1, H]
#         assert context_output.shape[2] == L1

#         start_logits = self.entity_start_classifier(context_output)  # [B, num_labels, L1, 1]
#         end_logits = self.entity_end_classifier(context_output)  # [B, num_labels, L1, 1]
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         start_logits = start_logits.transpose(1, 2)
#         end_logits = end_logits.transpose(1, 2)
#         return start_logits, end_logits  # [B, L1, num_labels]



# class LabelEnhanceBert(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         if pretrained_model == 'microsoft/deberta-v3-large':
#             self.bert = DebertaV2Model.from_pretrained(pretrained_model)
#         elif pretrained_model == 'bert-large-uncased':
#             self.bert = BertModel.from_pretrained(pretrained_model)
#         self.config = self.bert.config
#         # self.pos_embedding = nn.Embedding(19, 512)
#         # self.ent_embedding = nn.Embedding(19, 512)
#         self.dropout = nn.Dropout(p=0.1)
#         # self.fc_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         # self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         # self.fc_3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
#         self.entity_start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
#         self.entity_end_classifier = nn.Linear(self.bert.config.hidden_size, 1)

#         self.attention_layer = BertLayer(self.config)
#         self.attention_layer2 = BertLayer(self.config)
        
#         # self._init_weights(self.pos_embedding)
#         # self._init_weights(self.ent_embedding)
#         # self.fc_1.apply(self._init_weights)
#         # self.fc_2.apply(self._init_weights)
#         # self.fc_3.apply(self._init_weights)
#         self.entity_start_classifier.apply(self._init_weights)
#         self.entity_end_classifier.apply(self._init_weights)
#         self.attention_layer.apply(self._init_weights)
#         self.attention_layer2.apply(self._init_weights)



#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


#     def forward(self, text_input_ids, text_attention_mask, text_token_type_ids, label_input_ids, 
#                 label_attention_mask, label_token_type_ids, text_pos_token=None, text_ent_token=None,
#                 label_pos_token=None, label_ent_token=None):
#         '''
#         text_input_ids: [B, L1]
#         text_attention_mask: [B, L1]
#         text_token_type_ids: [B, L1]
#         label_input_ids: [num_label, L2]
#         label_attention_mask: [num_label, L2]
#         label_token_type_ids: [num_label, L2]
#         text_pos_token: [B, L1]
#         text_ent_token: [B, L1]
#         label_pos_token: [num_label, L2]
#         label_ent_token: [num_label, L2]
#         '''
#         text_output = self.bert(
#             input_ids=text_input_ids,
#             attention_mask=text_attention_mask,
#             token_type_ids=text_token_type_ids
#         )
#         encode_text = text_output[0]  # [B, L1, H]
#         label_output = self.bert(
#             input_ids=label_input_ids,
#             attention_mask=label_attention_mask,
#             token_type_ids=label_token_type_ids
#         )
#         encode_label = label_output[0]  # [num_labels, L2, H]

#         encode_text = self.dropout(encode_text)
#         encode_label = self.dropout(encode_label)


#         B, L1, H = encode_text.shape
#         num_labels, L2, _ = encode_label.shape

#         label_knowledge_len = torch.sum(label_attention_mask, dim=-1)  # [num_labels]
#         label_knowledge_len = label_knowledge_len.tolist()  # [num_labels] list, 每一个label knowlwdge的长度
#         label_attention_mask = label_attention_mask.reshape(-1)  # [num_labels*L2]
#         encode_label = encode_label.reshape(-1, H)  # [num_labels*L2, H]
#         encode_label = encode_label[label_attention_mask == 1]
#         assert len(encode_label.shape) == 2
#         assert sum(label_knowledge_len) == encode_label.shape[0]

#         encode_label_split = torch.split(encode_label, label_knowledge_len, dim=0)  # list, 长度为num_labels, [L_label, H]

#         concatenate_text_label = []
#         max_len = L1 + L2

#         total_attention_mask = []
#         for i in range(B):
#             encode_text_i =  encode_text[i]  # [L1, H]
#             label_text = [torch.cat([label_split, encode_text_i], dim=0) for label_split in encode_label_split]  # 长度为num_labels的列表，下一步pad
#             all_pad_len = [max_len - l_t.shape[0] for l_t in label_text]
#             label_text = [torch.cat([label_text[j], torch.zeros((all_pad_len[j], H), dtype=torch.long, device=text_input_ids.device)], dim=0) for j in range(len(label_text))]
#             label_text = torch.stack(label_text, dim=0)  # [num_labels, L, H]
#             concatenate_text_label.append(label_text)
#             attention_mask_i = [[1] * (max_len - l) + [0] * l for l in all_pad_len]
#             assert len(attention_mask_i) == num_labels
#             total_attention_mask.append(attention_mask_i)
            
        
#         total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long, device=text_input_ids.device)  # [B, num_labels, L]
        
#         assert len(concatenate_text_label) == B == len(total_attention_mask)
#         concatenate_text_label = torch.stack(concatenate_text_label, dim=0)  # [B, num_labels, L, H]
#         assert concatenate_text_label.shape[1] == num_labels == total_attention_mask.shape[1]
#         assert concatenate_text_label.shape[2] == max_len == total_attention_mask.shape[-1]
#         assert concatenate_text_label.shape[3] == H
#         assert len(concatenate_text_label.shape) == 4 == len(total_attention_mask.shape) + 1


#         total_attention_mask = total_attention_mask.reshape(-1, max_len)  # [B*num_labels, L]
#         concatenate_text_label = concatenate_text_label.reshape(-1, max_len, H)  # [B*num_labels, L, H]

#         total_attention_mask = total_attention_mask.unsqueeze(-2).repeat(1, max_len, 1)  # [B*num_labels, L, L]
#         total_attention_mask = total_attention_mask.unsqueeze(1)
#         total_attention_mask = (1.0 - total_attention_mask) * (-100000.0)  # [B*num_labels, L, L]

#         seq_output = self.attention_layer(concatenate_text_label, total_attention_mask)  # [B*num_labels, L, H]
#         seq_output = seq_output.reshape(B, num_labels, max_len, H)

#         context_seq_output = []
#         for i in range(B):
#             seq_output_i = seq_output[i]  # [num_labels, L, H]
#             seq_output_i_tmp = []
#             for j in range(num_labels):
#                 assert label_knowledge_len[j] + L1 <= max_len
#                 seq_output_i_tmp.append(seq_output_i[j, label_knowledge_len[j]: label_knowledge_len[j] + L1, :])
#             seq_output_i_tmp = torch.stack(seq_output_i_tmp, dim=0)  # [num_labels, L1, H]
#             context_seq_output.append(seq_output_i_tmp)
        
#         context_seq_output = torch.stack(context_seq_output, dim=0)  # [B, num_labels, L1, 1]

#         assert context_seq_output.shape[1] == num_labels

#         start_logits = self.entity_start_classifier(context_seq_output)  # [B, num_labels, L1, 1]
#         end_logits = self.entity_end_classifier(context_seq_output)  # [B, num_labels, L1, 1]
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         start_logits = start_logits.transpose(1, 2)
#         end_logits = end_logits.transpose(1, 2)
#         return start_logits, end_logits  # [B, L1, num_labels]


#         encode_text = encode_text.unsqueeze(1).repeat(1, num_labels, 1, 1)  # [B, L1, H]->[B, 1, L1, H]->[B, num_labels, L1, H]
#         encode_label = encode_label.unsqueeze(0).repeat(B, 1, 1, 1)  # [num_labels, L2, H]->[1, num_labels, L2, H]->[B, num_labels, L2, H]

#         label_attn_mask_extend = label_attention_mask.unsqueeze(0).repeat(B, 1, 1)  # [num_labels, L2]->[1, num_labels, L2]->[B, num_labels, L2]
#         text_attn_mask_extend = text_attention_mask.unsqueeze(1).repeat(1, num_labels, 1)  # [B, L1]->[B, 1, L1]->[B, num_labels, L1]

#         attention_mask_extend = torch.cat([label_attn_mask_extend, text_attn_mask_extend], dim=-1)  # [B, num_labels, L1+L2]
#         attention_mask_extend = attention_mask_extend.unsqueeze(-1).repeat(1, 1, 1, L1+L2)  # [B, num_labels, L1+L2]->[B, num_labels, L1+L2, L1+L2]
#         attention_mask_extend = attention_mask_extend.reshape(B * num_labels, L1+L2, L1+L2)  # [B*num_labels, L1+L2, L1+L2]
#         attention_mask_extend = attention_mask_extend.unsqueeze(1)
#         attention_mask_extend = (1.0 - attention_mask_extend) * (-100000.0)

#         hidden_state_extend = torch.cat([encode_label, encode_text], dim=-2)  # [B, num_labels, L1+L2, H]
#         hidden_state_extend = hidden_state_extend.reshape(B * num_labels, L1+L2, H)  # [B*num_labels, L1+L2, H]


#         seq_output = self.attention_layer(hidden_state_extend, attention_mask_extend)  # [B*num_labels, L1+L2, H]
#         seq_output = self.attention_layer2(seq_output, attention_mask_extend)  # [B*num_labels, L1+L2, H]
#         seq_output = seq_output.reshape(B, num_labels, L1+L2, H)
#         context_output = seq_output[:, :, L2:, :]  # [B, num_labels, L2, H]
#         assert context_output.shape[2] == L1

#         start_logits = self.entity_start_classifier(context_output)  # [B, num_labels, L1, 1]
#         end_logits = self.entity_end_classifier(context_output)  # [B, num_labels, L1, 1]
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         start_logits = start_logits.transpose(1, 2)
#         end_logits = end_logits.transpose(1, 2)
#         return start_logits, end_logits  # [B, L1, num_labels]






























































# class LabelEnhanceBert(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = DebertaV2Model.from_pretrained(pretrained_model)
#         # self.pos_embedding = nn.Embedding(19, 512)
#         # self.ent_embedding = nn.Embedding(19, 512)
#         self.dropout = nn.Dropout(p=0.1)
#         self.fc_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         self.fc_2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
#         self.fc_3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
#         self.entity_start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
#         self.entity_end_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
#         # self._init_weights(self.pos_embedding)
#         # self._init_weights(self.ent_embedding)
#         self._init_weights(self.fc_1)
#         self._init_weights(self.fc_2)
#         self._init_weights(self.fc_3)
#         self._init_weights(self.entity_start_classifier)
#         self._init_weights(self.entity_end_classifier)



#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


#     def forward(self, text_input_ids, text_attention_mask, text_token_type_ids, label_input_ids, 
#                 label_attention_mask, label_token_type_ids, text_pos_token=None, text_ent_token=None,
#                 label_pos_token=None, label_ent_token=None):
#         '''
#         text_input_ids: [B, L1]
#         text_attention_mask: [B, L1]
#         text_token_type_ids: [B, L1]
#         label_input_ids: [num_label, L2]
#         label_attention_mask: [num_label, L2]
#         label_token_type_ids: [num_label, L2]
#         text_pos_token: [B, L1]
#         text_ent_token: [B, L1]
#         label_pos_token: [num_label, L2]
#         label_ent_token: [num_label, L2]
#         '''
#         text_output = self.bert(
#             input_ids=text_input_ids,
#             attention_mask=text_attention_mask,
#             token_type_ids=text_token_type_ids
#         )
#         encode_text = text_output[0]  # [B, L1, H]
#         label_output = self.bert(
#             input_ids=label_input_ids,
#             attention_mask=label_attention_mask,
#             token_type_ids=label_token_type_ids
#         )
#         encode_label = label_output[0]  # [num_labels, L2, H]
#         # text_pos_embed = self.pos_embedding(text_pos_token)
#         # label_pos_embed = self.pos_embedding(label_pos_token)
#         # text_ent_embed = self.ent_embedding(text_ent_token)
#         # label_ent_embed = self.ent_embedding(label_ent_token)
        
#         # encode_text = torch.cat([encode_text, text_pos_embed, text_ent_embed], dim=-1)
#         # encode_label = torch.cat([encode_label, label_pos_embed, label_ent_embed], dim=-1)
#         encode_text = self.dropout(encode_text)
#         encode_label = self.dropout(encode_label)


#         B, L1, H = encode_text.shape
#         num_labels, L2, _ = encode_label.shape
#         token_feature = self.fc_1(encode_text)  # [B, L1, H]
#         label_feature = self.fc_2(encode_label)  # [num_labels, L2, H]
#         # label_feature_t = label_feature.permute(2, 0, 1).view(hidden_size, -1)  # [H, num_label * L2]
#         label_feature_t = label_feature.permute(2, 0, 1).reshape([H, -1])  # [num_labels, L2, H]->[H, num_labels * L2]
#         # scores = torch.matmul(token_feature, label_feature_t).view(
#         #     batch_size,
#         #     context_seq_len,
#         #     num_label,
#         #     -1
#         # )  # [B, L1, num_label, L2]
#         scores = torch.matmul(token_feature, label_feature_t).reshape(
#             B, L1, num_labels, -1
#         )  # [B, L1, H]*[H, num_labels*L2]=[B, L1, num_labels * L2]->[B, L1, num_labels, L2]
#         extend_label_attention_mask = label_attention_mask[None, None, :, :]  # [1, 1, num_labels, L2]
#         extend_label_attention_mask = (1.0 - extend_label_attention_mask) * (-10000.0)
#         scores = scores + extend_label_attention_mask
#         scores = F.softmax(scores, dim=-1)  # [B, L1, num_label, L2]
#         weight_label_feature = label_feature.unsqueeze(0).unsqueeze(0).expand([B, L1, num_labels, L2, H]) \
#             * scores.unsqueeze(-1)  # [B, L1, num_label, L2, H] * [B, L1, num_label, L2, H]
#         token_feature = token_feature.unsqueeze(2).expand([B, L1, num_labels, H])  # [B, L1, num_label, H]
#         weight_label_feature_sum = torch.sum(weight_label_feature, dim=-2)  # [B, L1, num_label, H]
#         fused_feature = token_feature + weight_label_feature_sum  # [B, L1, num_label, H]
#         fused_feature = torch.tanh(self.fc_3(fused_feature))  # [B, L1, num_label, H]
#         start_logits = self.entity_start_classifier(fused_feature)  # [B, L1, num_label, 1]
#         end_logits = self.entity_end_classifier(fused_feature)  # [B, L1, num_label, 1]
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         return start_logits, end_logits  # [B, L1, num_label]

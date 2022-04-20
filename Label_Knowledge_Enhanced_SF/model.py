import torch
import torch.nn as nn
from transformers import BertModel
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



# class BertTagger(nn.Module):
#     def __init__(self, pretrained_model):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(pretrained_model)
#         self.dropout = nn.Dropout(p=0.1)
#         self.output_layer = nn.Linear(self.bert.config.hidden_size, 2)
#         self.output_layer.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
#         self.output_layer.bias.data.zero_()
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         bert_outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         last_hidden_state = bert_outputs.last_hidden_state  # [B, L, H]
#         last_hidden_state = self.dropout(last_hidden_state)
#         io_logits = self.output_layer(last_hidden_state)  # [B, L, 2]
#         return io_logits


class LabelEnhanceBert(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pos_embedding = nn.Embedding(19, 512)
        self.ent_embedding = nn.Embedding(19, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_1 = nn.Linear(self.bert.config.hidden_size + 1024, self.bert.config.hidden_size + 1024, bias=False)
        self.fc_2 = nn.Linear(self.bert.config.hidden_size + 1024, self.bert.config.hidden_size + 1024, bias=False)
        self.fc_3 = nn.Linear(self.bert.config.hidden_size + 1024, self.bert.config.hidden_size + 1024)
        self.entity_start_classifier = nn.Linear(self.bert.config.hidden_size + 1024, 1)
        self.entity_end_classifier = nn.Linear(self.bert.config.hidden_size + 1024, 1)
        
        # self._init_weights(self.pos_embedding)
        # self._init_weights(self.ent_embedding)
        # self._init_weights(self.fc_1)
        # self._init_weights(self.fc_2)
        # self._init_weights(self.fc_3)
        # self._init_weights(self.entity_start_classifier)
        # self._init_weights(self.entity_end_classifier)



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
                label_attention_mask, label_token_type_ids, text_pos_token, text_ent_token,
                label_pos_token, label_ent_token):
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
        text_pos_embed = self.pos_embedding(text_pos_token)
        label_pos_embed = self.pos_embedding(label_pos_token)
        text_ent_embed = self.ent_embedding(text_ent_token)
        label_ent_embed = self.ent_embedding(label_ent_token)
        
        encode_text = torch.cat([encode_text, text_pos_embed, text_ent_embed], dim=-1)
        encode_label = torch.cat([encode_label, label_pos_embed, label_ent_embed], dim=-1)
        encode_text = self.dropout(encode_text)
        encode_label = self.dropout(encode_label)


        B, L1, H = encode_text.shape
        num_labels, L2, _ = encode_label.shape
        token_feature = self.fc_1(encode_text)  # [B, L1, H]
        label_feature = self.fc_2(encode_label)  # [num_labels, L2, H]
        # label_feature_t = label_feature.permute(2, 0, 1).view(hidden_size, -1)  # [H, num_label * L2]
        label_feature_t = label_feature.permute(2, 0, 1).reshape([H, -1])  # [num_labels, L2, H]->[H, num_labels * L2]
        # scores = torch.matmul(token_feature, label_feature_t).view(
        #     batch_size,
        #     context_seq_len,
        #     num_label,
        #     -1
        # )  # [B, L1, num_label, L2]
        scores = torch.matmul(token_feature, label_feature_t).reshape(
            B, L1, num_labels, -1
        )  # [B, L1, H]*[H, num_labels*L2]=[B, L1, num_labels * L2]->[B, L1, num_labels, L2]
        extend_label_attention_mask = label_attention_mask[None, None, :, :]  # [1, 1, num_labels, L2]
        extend_label_attention_mask = (1.0 - extend_label_attention_mask) * (-10000.0)
        scores = scores + extend_label_attention_mask
        scores = F.softmax(scores, dim=-1)  # [B, L1, num_label, L2]
        weight_label_feature = label_feature.unsqueeze(0).unsqueeze(0).expand([B, L1, num_labels, L2, H]) \
            * scores.unsqueeze(-1)  # [B, L1, num_label, L2, H] * [B, L1, num_label, L2, H]
        token_feature = token_feature.unsqueeze(2).expand([B, L1, num_labels, H])  # [B, L1, num_label, H]
        weight_label_feature_sum = torch.sum(weight_label_feature, dim=-2)  # [B, L1, num_label, H]
        fused_feature = token_feature + weight_label_feature_sum  # [B, L1, num_label, H]
        fused_feature = torch.tanh(self.fc_3(fused_feature))  # [B, L1, num_label, H]
        start_logits = self.entity_start_classifier(fused_feature)  # [B, L1, num_label, 1]
        end_logits = self.entity_end_classifier(fused_feature)  # [B, L1, num_label, 1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits  # [B, L1, num_label]


# class Classifier_Layer(nn.Module):
#     def __init__(self, num_label, out_features, bias=True):
#         super(Classifier_Layer, self).__init__()
#         self.num_label = num_label
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(num_label, out_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(num_label))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()


#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)

#     def forward(self, input):
#         x = torch.mul(input, self.weight)  # [B, L1, num_label, H]
#         x = torch.sum(x, -1)  # [-1, class_num]
#         if self.bias is not None:
#             x = x + self.bias
#         return x  # [B, L1, num_label]
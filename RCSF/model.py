# import torch
from torch import nn
from transformers import BertModel
# import torch.nn.functional as F

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

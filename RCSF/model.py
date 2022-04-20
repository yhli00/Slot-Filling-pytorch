from torch import nn
from transformers import BertModel, BertPreTrainedModel

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

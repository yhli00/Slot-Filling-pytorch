import torch
# from paddle.optimizer import AdamW
# from paddlenlp.transformers import LinearDecayWithWarmup
# from paddle.io import DataLoader
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer, BertTokenizer
# import paddle.nn as nn
import torch.nn as nn
# import paddle
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from model import LabelEnhanceBert
import logging
from dataset import get_dataset, collate_fn
from tqdm import tqdm
import os
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = LabelEnhanceBert(self.args.pretrained_model)
        if self.args.pretrained_model == 'deepset/bert-large-uncased-whole-word-masking-squad2':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') 
        if self.args.pretrained_model == 'bert-large-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        if self.args.pretrained_model == 'bert-large-uncased-whole-word-masking':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        if self.args.pretrained_model == 'microsoft/deberta-v3-large':
            self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
        train_datasets, valid_dataset, test_dataset = get_dataset(
            self.args.target_domain,
            self.args.n_samples,
            self.tokenizer,
            context_max_len=self.args.context_max_len,
            label_max_len=self.args.label_max_len
        )
        self.train_datas = []
        for dataset in train_datasets:
            self.train_datas.append(
                {
                    'domain_name': dataset.domain_name,
                    'label_knowleges_input_ids': dataset.label_knowlege_token,
                    'label_knowleges_attention_mask': dataset.label_knowlege_attention_mask,
                    'label_knowleges_token_type_ids': dataset.label_knowlege_token_type_ids,
                    'label_knowleges_pos_token': dataset.label_knowlege_pos_token,
                    'label_knowleges_ent_token': dataset.label_knowlege_ent_token,
                    'num_labels': dataset.num_labels,
                    'dataloader': DataLoader(
                        dataset,
                        shuffle=True,
                        batch_size=self.args.batch_size,
                        num_workers=self.args.num_workers,
                        collate_fn=collate_fn
                    )
                }
            )
        self.valid_data = {
            'domain_name': valid_dataset.domain_name,
            'dataloader': DataLoader(valid_dataset, shuffle=False, batch_size=4, 
                                     num_workers=self.args.num_workers, collate_fn=collate_fn),
            'label_knowleges_input_ids': valid_dataset.label_knowlege_token,
            'label_knowleges_attention_mask': valid_dataset.label_knowlege_attention_mask,
            'label_knowleges_token_type_ids': valid_dataset.label_knowlege_token_type_ids,
            'label_knowleges_pos_token': valid_dataset.label_knowlege_pos_token,
            'label_knowleges_ent_token': valid_dataset.label_knowlege_ent_token,
            'num_labels': valid_dataset.num_labels
        }
        
        self.test_data = {
            'domain_name': test_dataset.domain_name,
            'dataloader': DataLoader(test_dataset, shuffle=False, batch_size=4, 
                                     num_workers=self.args.num_workers, collate_fn=collate_fn),
            'label_knowleges_input_ids': test_dataset.label_knowlege_token,
            'label_knowleges_attention_mask': test_dataset.label_knowlege_attention_mask,
            'label_knowleges_token_type_ids': test_dataset.label_knowlege_token_type_ids,
            'label_knowleges_pos_token': test_dataset.label_knowlege_pos_token,
            'label_knowleges_ent_token': test_dataset.label_knowlege_ent_token,
            'num_labels': test_dataset.num_labels
        }

        
        self.optimizer, self.lr_scheduler = self._get_optimizer_and_scheduler()
        self.loss_fn = self._get_loss_function()
        self.device = args.device
        self.model = self.model.to(self.device)


    def _get_optimizer_and_scheduler(self):
        # no_decay = ["bias", "LayerNorm.weight"]  # 如果是bias或者LayerNorm.weight，weight_decay就是0
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        train_data_length = 0
        for train_data in self.train_datas:
            train_data_length += len(train_data['dataloader'])
        assert train_data_length != 0
        t_total = (train_data_length // (self.args.gradient_accumulation_steps) + 1) * self.args.num_epochs  # accumulate_grad_batches默认是1
        # lr_scheduler = LinearDecayWithWarmup(
        #     self.args.lr,
        #     total_steps=t_total,
        #     warmup=self.args.warmup_rate
        # )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
            betas=(0.9, 0.98)
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_rate * t_total),
            num_training_steps=t_total
        )
        return optimizer, lr_scheduler


    def _get_loss_function(self):
        return nn.BCEWithLogitsLoss(reduction='none')





    def train(self):
        early_stop_cnt = 0
        max_f1 = -1
        early_stop_flag = False
        for epoch in range(self.args.num_epochs):
            for train_data in self.train_datas:
                domain_name = train_data['domain_name']
                train_dataloader = train_data['dataloader']
                train_loss = 0.0  # 这个domain的train_loss
                for step, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{self.args.num_epochs}, Domain: {domain_name}', ncols=90)):
                    self.model.train()
                    self.optimizer.zero_grad()
                    
                    label_knowleges_input_ids = train_data['label_knowleges_input_ids'].to(self.device)  # [num_labels, L2]
                    label_knowleges_attention_mask = train_data['label_knowleges_attention_mask'].to(self.device)  # [num_labels, L2]
                    label_knowleges_token_type_ids = train_data['label_knowleges_token_type_ids'].to(self.device)  # [num_labels, L2]
                    num_labels = train_data['num_labels']
                    # label_knowleges_pos_token = train_data['label_knowleges_pos_token'].to(self.device)
                    # label_knowleges_ent_token = train_data['label_knowleges_ent_token'].to(self.device)

                    input_ids = batch['input_ids'].to(self.device)  # [B, L1]
                    attention_mask = batch['attention_mask'].to(self.device)  # [B, L1]
                    token_type_ids = batch['token_type_ids'].to(self.device)  # [B, L1]
                    start_labels = batch['start_labels'].to(self.device)  # [B, num_labels, L1]
                    end_labels = batch['end_labels'].to(self.device)  # [B, num_labels, L1]
                    start_token_mask = batch['start_token_mask'].to(self.device)  # [B, L1]
                    end_token_mask = batch['end_token_mask'].to(self.device)  # [B, L1]
                    # text_pos_token = batch['pos_token'].to(self.device)
                    # text_ent_token = batch['ent_token'].to(self.device)

                    start_logits, end_logits = self.model(
                        text_input_ids=input_ids,
                        text_attention_mask=attention_mask,
                        text_token_type_ids=token_type_ids,
                        label_input_ids=label_knowleges_input_ids,
                        label_attention_mask=label_knowleges_attention_mask,
                        label_token_type_ids=label_knowleges_token_type_ids
                        # text_pos_token=text_pos_token,
                        # text_ent_token=text_ent_token,
                        # label_pos_token=label_knowleges_pos_token,
                        # label_ent_token=label_knowleges_ent_token
                    )  # [B, L1, num_labels]
                    
                    B, L1, _ = start_logits.shape

                    start_token_mask = start_token_mask.unsqueeze(-2).repeat(1, num_labels, 1)  # [B, num_labels, L1]
                    end_token_mask = end_token_mask.unsqueeze(-2).repeat(1, num_labels, 1)  # [B, num_labels, L1]

                    start_logits = start_logits.transpose(1, 2)  # [B, num_labels, L1]
                    end_logits = end_logits.transpose(1, 2)  # [B, num_labels, L1]

                    start_loss = self.loss_fn(start_logits.reshape(-1), start_labels.reshape(-1).float())
                    start_loss = (start_loss * start_token_mask.reshape(-1)).sum() / start_token_mask.float().sum()
                    end_loss = self.loss_fn(end_logits.reshape(-1), end_labels.reshape(-1).float())
                    end_loss = (end_loss * end_token_mask.reshape(-1)).sum() / end_token_mask.float().sum()

                    total_loss = start_loss + end_loss

                    with torch.no_grad():
                        train_loss += total_loss.item()

                    if self.args.gradient_accumulation_steps > 1:
                        total_loss = total_loss / self.args.gradient_accumulation_steps
                    total_loss.backward()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()

                acc, recall, f1 = self.evaluate('valid')
                logger.info(f'Epoch {epoch + 1}, Domain {domain_name}, train_loss {train_loss / len(train_data):.6f},    Evalution acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')

                if f1 > max_f1:
                    max_f1 = f1
                    early_stop_cnt = 0
                    self.save_model(epoch + 1, max_f1)
                else:
                    early_stop_cnt += 1
                if early_stop_cnt >= self.args.early_stop:
                    logger.info(f'Reach early stop count {self.args.early_stop}, training stop')
                    early_stop_flag = True
                    break
            if early_stop_flag:
                break


    def evaluate(self, mode):
        if mode == 'valid':
            valid_data = self.valid_data
        if mode == 'test':
            valid_data = self.test_data
        
        pred_spans = []
        all_context_srcs = []
        all_context_ids = []
        all_token_to_origin_index = []
        all_pred_tags = []
        all_gold_tags = []
        all_label_srcs = []
        self.model.eval()
        dataloader = valid_data['dataloader']
        num_labels = valid_data['num_labels']
        for batch in tqdm(dataloader, desc='Evaluating', ncols=90):
            input_ids = batch['input_ids'].to(self.device)  # [B, L1]
            attention_mask = batch['attention_mask'].to(self.device)  # [B, L1]
            token_type_ids = batch['token_type_ids'].to(self.device)  # [B, L1]
            start_token_mask = batch['start_token_mask'].detach().cpu().numpy()  # [B, L1]
            end_token_mask = batch['end_token_mask'].detach().cpu().numpy()  # [B, L1]
            # text_pos_token = batch['pos_token'].to(self.device)
            # text_ent_token = batch['ent_token'].to(self.device)

            context_srcs = batch['context_src']  # [B]str
            context_ids = batch['context_id']  # [B]int
            tags = batch['tags']  # [B, num_labels]str
            token_to_origin_index = batch['context_token_to_origin_index']  # [B] list
            label_srcs = batch['label_src']  # [B]str

            label_knowleges_input_ids = valid_data['label_knowleges_input_ids'].to(self.device)  # [num_labels, L2]
            label_knowleges_attention_mask = valid_data['label_knowleges_attention_mask'].to(self.device)  # [num_labels, L2]
            label_knowleges_token_type_ids = valid_data['label_knowleges_token_type_ids'].to(self.device)  # [num_labels, L2]            
            # label_knowleges_pos_token = valid_data['label_knowleges_pos_token'].to(self.device)
            # label_knowleges_ent_token = valid_data['label_knowleges_ent_token'].to(self.device)


            all_context_srcs.extend(context_srcs)
            all_context_ids.extend(context_ids)
            all_token_to_origin_index.extend(token_to_origin_index)
            all_label_srcs.extend(label_srcs)

            with torch.no_grad():
                start_logits, end_logits = self.model(
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                    text_token_type_ids=token_type_ids,
                    label_input_ids=label_knowleges_input_ids,
                    label_attention_mask=label_knowleges_attention_mask,
                    label_token_type_ids=label_knowleges_token_type_ids
                    # text_pos_token=text_pos_token,
                    # text_ent_token=text_ent_token,
                    # label_pos_token=label_knowleges_pos_token,
                    # label_ent_token=label_knowleges_ent_token
                )  # [B, L1, num_labels]
            
            start_logits = start_logits.transpose(1, 2)  # [B, num_labels, L1]
            end_logits = end_logits.transpose(1, 2)  # [B, num_labels, L1]
            start_preds = F.softmax(start_logits, dim=-1)  # [B, num_labels, L1]
            start_preds = start_preds.detach().cpu().numpy()
            end_preds = F.softmax(end_logits, dim=-1)  # [B, num_labels, L1]
            end_preds = end_preds.detach().cpu().numpy()

            
            for i in range(len(start_logits)):
                spans_pro = []
                for j in range(num_labels):
                    start_pred = start_preds[i][j]  # [L1]
                    end_pred = end_preds[i][j]  # [L1]
                    start_sorted = np.argsort(-start_pred)  # 返回的是从大到小的index
                    end_sorted = np.argsort(-end_pred)
                    for start in start_sorted[:self.args.n_top]:
                        if start_token_mask[i][start] == 0:
                            break
                        for end in end_sorted[:self.args.n_top]:
                            if end_token_mask[i][end] == 0:
                                break
                            if start <= end and end - start < 8:
                                spans_pro.append((tags[i][j], start, end, start_pred[start] + end_pred[end]))
                pred_spans.append(spans_pro)
        t_pred = {}
        t_gold = {}
        contexts = {}


        for idx, pred_span, label_src, context_src, token_to_origin_index in zip(
            all_context_ids,
            pred_spans,
            all_label_srcs,
            all_context_srcs,
            all_token_to_origin_index
        ):
            t_pred.setdefault(idx, list())
            t_gold[idx] = label_src.split()
            contexts[idx] = context_src
            for tag, start, end, p in pred_span:
                t_pred[idx].append((tag, (start, end), token_to_origin_index, p))
        
        for idx in t_pred:
            elem_pred = t_pred[idx]
            elem_pred = sorted(elem_pred, key=lambda x: x[-1], reverse=True)  # 根据P_start+P_end从大到小排序
            elem_pred = self._remove_overlap(elem_pred)
            context = contexts[idx]
            pred_tag = ['O'] * len(context.split())
            elem_pred = sorted(elem_pred, key=lambda x: x[1][1])  # 根据start从小到大排序
            for tag, (start, end), token_to_origin_index, _ in elem_pred:
                origin_start = token_to_origin_index[start]
                origin_end = token_to_origin_index[end]
                pred_tag[origin_start] = 'B-' + tag
                for i in range(origin_start + 1, origin_end + 1):
                    pred_tag[i] = 'I-' + tag
            all_pred_tags.append(pred_tag)
            all_gold_tags.append(t_gold[idx])
        
        acc = precision_score(all_gold_tags, all_pred_tags)
        recall = recall_score(all_gold_tags, all_pred_tags)
        f1 = f1_score(all_gold_tags, all_pred_tags)

        if mode == 'test':
            logger.info('***********************************************************')
            logger.info(f'Target domain = {self.args.target_domain}')
            logger.info(f'Test result: acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            logger.info('***********************************************************')
        return acc, recall, f1




    def _remove_overlap(self, spans):
        """
        remove overlapped spans greedily for flat-ner
        Args:
            spans: list of tuple (start, end), which means [start, end] is a ner-span
        Returns:
            spans without overlap
        """
        output = []
        occupied = set()
        for span in spans:
            _, (start, end), _, _ = span
            is_occupied = False
            for x in range(start, end + 1):
                if x in occupied:
                    is_occupied = True
                    break
            if is_occupied:
                continue
            output.append(span)
            for x in range(start, end + 1):
                occupied.add(x)
        return output


    # def save_model(self, epoch, domain_name):
    #     if not os.path.exists(os.path.join(self.args.model_dir, domain_name, str(self.args.n_samples))):
    #         os.makedirs(os.path.join(self.args.model_dir, domain_name, str(self.args.n_samples)))
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'args': self.args
    #     }
    #     torch.save(checkpoint, os.path.join(self.args.model_dir, domain_name, str(self.args.n_samples), 'model.pth'))
    #     logger.info("Saved model checkpoint to %s", os.path.join(self.args.model_dir, domain_name, str(self.args.n_samples), 'model.pth'))


    def save_model(self, epoch, max_f1):
        if not os.path.exists(os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples))):
            os.makedirs(os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'args': self.args,
            'max_f1': max_f1
        }
        torch.save(checkpoint, os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples), 'model.pth'))
        logger.info("Saved model checkpoint to %s", os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples), 'model.pth'))
    

def load_model(model_dir, target_domain, n_samples):
    checkpoint = torch.load(os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
    args = checkpoint['args']
    trainer = Trainer(args)
    # if args.num_gpus == 2:
    #     trainer.model = nn.DataParallel(trainer.model.to('cuda'), device_ids=[0])
    #     trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
    #     trainer.model = trainer.model.module
    # else:
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Model loaded from %s', os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
    return trainer
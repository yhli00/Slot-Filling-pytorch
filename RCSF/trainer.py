import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer, RobertaTokenizer
from model import BertMRC
import logging
from dataset import get_dataset, collate_fn
from tqdm import tqdm
import os
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, args):
        self.args = args
        self.model = BertMRC(self.args.pretrained_model)
        if self.args.pretrained_model == 'deepset/bert-large-uncased-whole-word-masking-squad2':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') 
        if self.args.pretrained_model == 'bert-large-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        if self.args.pretrained_model == 'bert-large-uncased-whole-word-masking':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        if self.args.pretrained_model == 'roberta-large':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        train_dataset, valid_datset, test_dataset = get_dataset(
            self.args.target_domain,
            self.args.n_samples,
            self.tokenizer,
            max_len=self.args.max_len
        )
        self.train_data = DataLoader(train_dataset, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=collate_fn)
        self.valid_data = DataLoader(valid_datset, shuffle=False, batch_size=8, num_workers=self.args.num_workers, collate_fn=collate_fn)
        self.test_data = DataLoader(test_dataset, shuffle=False, batch_size=8, num_workers=self.args.num_workers, collate_fn=collate_fn)

        self.optimizer, self.scheduler = self._get_optimizer_and_schedule()
        self.loss_fn = self._get_loss_function()

        if self.args.num_gpus == 1:
            self.device = torch.device('cuda')
        self.model = self.model.to(self.device)


    def _get_optimizer_and_schedule(self):
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
        optimizer = AdamW(
            self.model.parameters(),
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon
        )
        t_total = (len(self.train_data) // (self.args.gradient_accumulation_steps * self.args.num_gpus) + 1) * self.args.num_epochs  # accumulate_grad_batches默认是1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.args.lr,
            pct_start=self.args.warmup_rate,
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, 
            anneal_strategy='linear'  # 退火策略
        )
        return optimizer, scheduler  # interval指的是每一个step执行一次scheduler.step()
    
    def _get_loss_function(self):
        return nn.BCEWithLogitsLoss(reduction='none')
        # return nn.CrossEntropyLoss(reduction='mean')
    

    def train(self):
        early_stop_cnt = 0
        max_f1 = -1

        for epoch in range(self.args.num_epochs):
            train_loss = 0.0
            for step, batch in enumerate(tqdm(self.train_data, desc=f'Train Epoch {epoch + 1}/{self.args.num_epochs}')):
                self.model.train()
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)  # [B, seq_len]
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                start_token_mask = batch['start_token_mask'].to(self.device)  # [B, seq_len]
                end_token_mask = batch['end_token_mask'].to(self.device)
                start_labels = batch['start_labels'].to(self.device)  # [B, L]
                end_labels = batch['end_labels'].to(self.device)  # [B, L]
                # start_label = batch['start_label'].to(self.device)  # [B]
                # end_label = batch['end_label'].to(self.device)

                input_span_mask = batch['input_span_mask'].to(self.device)

                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    input_span_mask=input_span_mask
                )  # [B, L]

                # start_loss = self.loss_fn(start_logits, start_label)
                # end_loss = self.loss_fn(end_logits, end_label)
                # total_loss = start_loss + end_loss
                start_loss = self.loss_fn(start_logits.reshape(-1), start_labels.reshape(-1).float())
                start_loss = (start_loss * start_token_mask.reshape(-1)).sum() / start_token_mask.sum()
                end_loss = self.loss_fn(end_logits.reshape(-1), end_labels.reshape(-1).float())
                end_loss = (end_loss * end_token_mask.reshape(-1)).sum() / end_token_mask.sum()
                total_loss = start_loss + end_loss

                with torch.no_grad():
                    train_loss += total_loss.item()

                if self.args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.args.gradient_accumulation_steps
                total_loss.backward()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
            with torch.no_grad():
                acc, recall, f1 = self.evaluate('valid')
            # logger.info(f'Epoch {epoch + 1}, train_loss {train_loss / len(self.train_data):.6f}, MRC acc = {acc: .4f}, Evalution acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            logger.info(f'Epoch {epoch + 1}, train_loss {train_loss / len(self.train_data):.6f}, Evalution acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            if f1 > max_f1:
                max_f1 = f1
                early_stop_cnt = 0
                self.save_model(epoch + 1)
            else:
                early_stop_cnt += 1
            if early_stop_cnt >= self.args.early_stop:
                logger.info(f'Reach early stop count {self.args.early_stop}, training stop')
                break
    

    def evaluate(self, mode):
        if mode == 'valid':
            valid_data = self.valid_data
        if mode == 'test':
            valid_data = self.test_data
        
        # total_num = 0.
        # right_num = 0.
        pred_spans = []
        # all_cls_logits = []
        all_tags = []
        all_context_srcs = []
        all_context_ids = []
        all_token_to_origin_index = []
        all_pred_tags = []
        all_gold_tags = []
        all_label_srcs = []
        self.model.eval()
        for batch in tqdm(valid_data, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)  # [B, seq_len]
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            start_token_mask = batch['start_token_mask'].detach().cpu().numpy()  # [B, seq_len]
            end_token_mask = batch['end_token_mask'].detach().cpu().numpy()

            context_srcs = batch['context_src']  # [B]str
            context_ids = batch['context_id']  # [B]int
            tags = batch['tag']  # [B]str
            token_to_origin_index = batch['token_to_origin_index'].detach().cpu().numpy()
            label_srcs = batch['label_src']  # [B]str

            input_span_mask = batch['input_span_mask'].to(self.device)
            # start_label = batch['start_label'].to(self.device)  # [B]
            # end_label = batch['end_label'].to(self.device)  # [B]
            

            all_context_srcs.extend(context_srcs)
            all_tags.extend(tags)
            all_context_ids.extend(context_ids)
            all_token_to_origin_index.extend(token_to_origin_index)
            all_label_srcs.extend(label_srcs)

            with torch.no_grad():
                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    input_span_mask=input_span_mask
                )  # [B, L]
            
            start_preds = torch.softmax(start_logits, -1).detach().cpu().numpy()  # [B, seq_len]
            end_preds = torch.softmax(end_logits, -1).detach().cpu().numpy()  # [B, seq_len]
            
            # start_preds = start_logits.detach().cpu().numpy()
            # end_preds = end_logits.detach().cpu().numpy()

            # with torch.no_grad():
            #     total_num += len(batch)
            #     start_hat = start_logits.argmax(dim=-1)  # [B]
            #     end_hat = end_logits.argmax(dim=-1)  # [B]
            #     start_right = start_hat == start_label
            #     end_right = end_hat == end_label
            #     right_num += torch.sum(start_right == end_right).detach().cpu().numpy()


            
            for i in range(len(start_preds)):
                spans_pro = []
                start_pred = start_preds[i]  # [seq_len]
                end_pred = end_preds[i]  # [seq_len]
                start_sorted = np.argsort(-start_pred)  # 返回的是从大到小的index
                end_sorted = np.argsort(-end_pred)
                # all_cls_logits.append(start_pred[0] + end_pred[0])
                for start in start_sorted[: self.args.n_top]:
                    for end in end_sorted[: self.args.n_top]:
                        if start_token_mask[i][start] == 0:
                            continue
                        if end_token_mask[i][end] == 0:
                            continue
                        if start > end:
                            continue
                        spans_pro.append((start, end, start_pred[start] + end_pred[end]))
                # for start in start_sorted[:self.args.n_top]:
                #     if start_token_mask[i][start] == 0:
                #         break
                #     for end in end_sorted[:self.args.n_top]:
                #         if end_token_mask[i][end] == 0:
                #             break
                #         if start <= end and end - start < 8:
                #             spans_pro.append((start, end, start_pred[start] + end_pred[end]))
                pred_spans.append(spans_pro)
            
        t_pred = {}
        t_gold = {}
        # t_cls = {}
        contexts = {}


        for idx, pred_span, label_src, context_src, token_to_origin_index, tag in zip(
            all_context_ids,
            pred_spans,
            all_label_srcs,
            all_context_srcs,
            all_token_to_origin_index,
            all_tags
        ):
            t_pred.setdefault(idx, list())
            t_gold[idx] = label_src.split()
            # if idx not in t_cls:
            #     t_cls[idx] = {}
            # t_cls[idx][tag] = cls_logits
            contexts[idx] = context_src
            for start, end, p in pred_span:
                t_pred[idx].append((tag, (start, end), token_to_origin_index, p))
        
        for idx in t_pred:
            elem_pred = t_pred[idx]
            elem_pred = sorted(elem_pred, key=lambda x: x[-1], reverse=True)  # 根据P_start+P_end从大到小排序
            elem_pred = self._remove_overlap(elem_pred)
            context = contexts[idx]
            pred_tag = ['O'] * len(context.split())
            elem_pred = sorted(elem_pred, key=lambda x: x[1][1])  # 根据start从小到大排序
            for tag, (start, end), token_to_origin_index, p in elem_pred:
                # if p < t_cls[idx][tag]:
                #     continue
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
        # acc = right_num / total_num

        if mode == 'test':
            logger.info('***********************************************************')
            logger.info(f'Target domain = {self.args.target_domain}')
            logger.info(f'Test result: acc {acc:.4f}, recall {recall:.4f}, f1 {f1:.4f}')
            # logger.info(f'MRC acc = {acc: .4f}')
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



    def save_model(self, epoch):
        if not os.path.exists(os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples))):
            os.makedirs(os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'args': self.args
        }
        torch.save(checkpoint, os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples), 'model.pth'))
        logger.info("Saved model checkpoint to %s", os.path.join(self.args.model_dir, self.args.target_domain, str(self.args.n_samples), 'model.pth'))
    
    @classmethod
    def load_model(cls, model_dir, target_domain, n_samples):
        checkpoint = torch.load(os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
        args = checkpoint['args']
        trainer = Trainer(args)
        if args.num_gpus == 2:
            trainer.model = nn.DataParallel(trainer.model.to('cuda'), device_ids=[0])
            trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
            trainer.model = trainer.model.module
        else:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Model loaded from %s', os.path.join(model_dir, target_domain, str(n_samples), 'model.pth'))
        return trainer
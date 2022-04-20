from utils import domain_set, slot2desp, domain2slots
import logging
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)

class DataProcessor():
    @staticmethod
    def read_file(file_path):
        sample_list = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                sample_list.append(line)
        logger.info("Loading data from {}, total nums is {}".format(file_path, len(sample_list)))
        return sample_list

    @staticmethod
    def get_all_data():
        data = {}
        for domain in domain_set:
            data[domain] = DataProcessor.read_file("data/snips/" + domain + '/' + domain + '.txt')
        data['atis'] = DataProcessor.read_file("data/atis/atis.txt")
        return data
    
    @staticmethod
    def convert_rowdata_to_mrc(all_data, dm_name, query_type):
        samples = []
        slot2query = {}
        if query_type == "desp":
            for slot, desp in slot2desp.items():
                slot2query[slot] = "what is the {}".format(desp)
            logger.info('Using queries from description.')
        context_id = 0
        for line in all_data:
            src, labels = line.strip().split("\t")
            char_label_list = [(char, label) for char, label in zip(src.split(), labels.split())]
            length = len(char_label_list)
            tags = []
            idx = 0
            while idx < length:
                _, label = char_label_list[idx]
                current_label = label[0]
                if current_label == "O":
                    idx += 1
                if current_label == "B":
                    end = idx + 1
                    while end < length and char_label_list[end][1][0] == "I":
                        end += 1
                    entity = " ".join(char_label_list[i][0] for i in range(idx, end))
                    tags.append(
                        {
                            'span': entity,
                            'tag': label[2:],
                            'begin': idx,
                            'end': end  # begin, end左闭右开
                        }
                    )
                    idx = end
            for tag_idx, (label, query) in enumerate(slot2query.items()):
                if label not in domain2slots[dm_name]:
                    continue
                samples.append(
                    {
                        'context_id': context_id,
                        'tag_id': tag_idx,
                        'context_src': src,
                        'label_src': labels,
                        'query_src': query,
                        'tag': label,
                        'start_positions': [tag['begin'] for tag in tags if tag['tag'] == label],
                        'end_positions': [tag['end'] - 1 for tag in tags if tag['tag'] == label],
                        'start_end_positions': [(tag['begin'], tag['end'] - 1) for tag in tags if tag['tag'] == label]
                    }
                )
            context_id += 1
        return samples


class MrcAndTagDataset(Dataset):
    def __init__(self, all_data, tokenizer, tag_schema='IO', max_len=128):
        super().__init__()
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.tag_schema = tag_schema
        self.max_len = max_len
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        data = self.all_data[index]
        context = data['context_src']
        query = data['query_src']
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token
        context = context.strip().split()
        query = query.strip().split()
        tokens = []
        start_token_mask = [0, 0]  # cls and sep
        end_token_mask = [0, 0]
        start_labels = [0, 0]
        end_labels = [0, 0]
        context_mask = [0, 0]
        context_token_to_origin_index = [0, 0]
        for word in query:
            token = self.tokenizer.tokenize(word)
            for _ in range(len(token)):
                context_mask.append(0)
                context_token_to_origin_index.append(0)
                start_token_mask.append(0)
                end_token_mask.append(0)
                start_labels.append(0)
                end_labels.append(0)
            tokens.extend(token)
        tokens = [cls_token] + tokens + [sep_token]
        token_type_ids = [0] * len(tokens)
        for idx, word in enumerate(context):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i, _ in enumerate(token):
                if i == 0:
                    start_token_mask.append(1)
                else:
                    start_token_mask.append(0)
                if i == len(token) - 1:
                    end_token_mask.append(1)
                else:
                    end_token_mask.append(0)
                if i == 0 and idx in data['start_positions']:
                    start_labels.append(1)
                else:
                    start_labels.append(0)
                if i == len(token) - 1 and idx in data['end_positions']:
                    end_labels.append(1)
                else:
                    end_labels.append(0)
                context_token_to_origin_index.append(idx)
                context_mask.append(1)
                token_type_ids.append(1)

        # truncate
        if len(tokens) > self.max_len - 1:
            tokens = tokens[:self.max_len - 1]
            context_mask = context_mask[:self.max_len - 1]
            context_token_to_origin_index = context_token_to_origin_index[:self.max_len - 1]
            start_labels = start_labels[:self.max_len - 1]
            end_labels = end_labels[:self.max_len - 1]
            start_token_mask = start_token_mask[:self.max_len - 1]
            end_token_mask = end_token_mask[:self.max_len - 1]
            token_type_ids = token_type_ids[:self.max_len - 1]
        
        tokens = tokens + [sep_token]
        context_mask = context_mask + [0]
        context_token_to_origin_index = context_token_to_origin_index + [0]
        start_labels = start_labels + [0]
        end_labels = end_labels + [0]
        start_token_mask = start_token_mask + [0]
        end_token_mask = end_token_mask + [0]
        token_type_ids = token_type_ids + [1]


        # pad
        token_len = len(tokens)
        attention_mask = [1] * token_len + [0] * (self.max_len - token_len)
        token_type_ids = token_type_ids + [0] * (self.max_len - token_len)
        for _ in range(self.max_len - token_len):
            tokens.append(pad_token)
            context_mask.append(0)
            context_token_to_origin_index.append(0)
            start_labels.append(0)
            end_labels.append(0)
            start_token_mask.append(0)
            end_token_mask.append(0)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return{
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_token_mask': torch.tensor(start_token_mask, dtype=torch.long),
            'end_token_mask': torch.tensor(end_token_mask, dtype=torch.long),
            'start_labels': torch.tensor(start_labels, dtype=torch.long),
            'end_labels': torch.tensor(end_labels, dtype=torch.long),
            'token_to_origin_index': torch.tensor(context_token_to_origin_index, dtype=torch.long),
            'context_id': data['context_id'],
            'tag_id': data['tag_id'],
            'label_src': data['label_src'],
            'query_src': data['query_src'],
            'context_src': data['context_src'],
            'tag': data['tag']
        }


def collate_fn(batch):  # batch是字典的列表
    output = {}
    output['input_ids'] = torch.stack([x['input_ids'] for x in batch])
    output['attention_mask'] = torch.stack([x['attention_mask'] for x in batch])
    output['token_type_ids'] = torch.stack([x['token_type_ids'] for x in batch])
    output['start_token_mask'] = torch.stack([x['start_token_mask'] for x in batch])
    output['end_token_mask'] = torch.stack([x['end_token_mask'] for x in batch])
    output['start_labels'] = torch.stack([x['start_labels'] for x in batch])
    output['end_labels'] = torch.stack([x['end_labels'] for x in batch])
    output['token_to_origin_index'] = torch.stack([x['token_to_origin_index'] for x in batch])
    output['context_id'] = []
    output['context_id'].extend([x['context_id'] for x in batch])
    output['tag_id'] = []
    output['tag_id'].extend([x['tag_id'] for x in batch])
    output['label_src'] = []
    output['label_src'].extend([x['label_src'] for x in batch])
    output['query_src'] = []
    output['query_src'].extend([x['query_src'] for x in batch])
    output['context_src'] = []
    output['context_src'].extend([x['context_src'] for x in batch])
    output['tag'] = []
    output['tag'].extend([x['tag'] for x in batch])
    return output


def get_dataset(tgt_domain, n_samples, tokenizer, max_len, query_type="desp", tag_schema='IO'):
    all_data = DataProcessor.get_all_data()
    train_data = []
    valid_data = []
    test_data = []
    train_data_orig = []
    valid_data_orig = []
    test_data_orig = []
    for dm_name, dm_data in all_data.items():  # dm_data是原始字符串的列表
        if dm_name != tgt_domain and dm_name != 'atis':  # atis只做测试用
            train_data.extend(DataProcessor.convert_rowdata_to_mrc(dm_data, dm_name, query_type=query_type))
            train_data_orig.extend(dm_data)
    
    train_data.extend(DataProcessor.convert_rowdata_to_mrc(all_data[tgt_domain][:n_samples], tgt_domain, query_type=query_type))  # 使用tgt_domain里的n_samples个训练样本
    train_data_orig.extend(all_data[tgt_domain][:n_samples])
    valid_data = DataProcessor.convert_rowdata_to_mrc(all_data[tgt_domain][n_samples:500], tgt_domain, query_type=query_type)
    valid_data_orig.extend(all_data[tgt_domain][n_samples:500])
    test_data = DataProcessor.convert_rowdata_to_mrc(all_data[tgt_domain][500:], tgt_domain, query_type=query_type)
    test_data_orig.extend(all_data[tgt_domain][500:])

    train_dataset = MrcAndTagDataset(train_data, tokenizer, tag_schema, max_len)
    valid_dataset = MrcAndTagDataset(valid_data, tokenizer, tag_schema, max_len)
    test_dataset = MrcAndTagDataset(test_data, tokenizer, tag_schema, max_len)

    logger.info(f'The target domain is {tgt_domain}')
    logger.info(f'train_data: {len(train_data_orig)} -> {len(train_data)}')
    logger.info(f'valid_data: {len(valid_data_orig)} -> {len(valid_data)}')
    logger.info(f'test_data: {len(test_data_orig)} -> {len(test_data)}')

    return train_dataset, valid_dataset, test_dataset



if __name__ == '__main__':
    from transformers import BertTokenizer
    # from tqdm import tqdm
    # import json
    from torch.utils.data import DataLoader
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    tgt_domain = 'BookRestaurant'
    n_samples = 0
    data = ['play a chant by mj cole	O O B-music_item O B-artist I-artist']
    all_data = DataProcessor.convert_rowdata_to_mrc(data, dm_name=tgt_domain, query_type='desp')
    # for data in all_data:
    #     print(data)
    dataset = MrcAndTagDataset(all_data, tokenizer, max_len=20)
    train_data = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn
    )
    for i in train_data:
        a = 1
    # print(len(dataset))
    # for i in dataset:
    #     print(i)
    # train_dataset, valid_dataset, test_dataset = get_dataset(tgt_domain, n_samples, tokenizer)
    # print(len(train_dataset))
    # print(len(valid_dataset))
    # print(len(test_dataset))
    # no_cnt = 0
    # multi_cnt = 0
    # single_cnt = 0
    # for i in tqdm(train_dataset):
    #     if len(i['start_end_positions']) == 0:
    #         no_cnt += 1
    #     if len(i['start_end_positions']) == 1:
    #         single_cnt += 1
    #     if len(i['start_end_positions']) > 1:
    #         multi_cnt += 1
    # print(no_cnt)
    # print(single_cnt)
    # print(multi_cnt)
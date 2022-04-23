from utils import domain_set, slot2desp
import logging
# import paddle
# import spacy
# from paddle.io import Dataset
from tqdm import tqdm
# import re
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

# POS_SET = ['NUM', 'ADP', 'PART', 'INTJ', 'SCONJ', 'ADV', 'SPACE', 'VERB', 'AUX', 'X', 
#            'CCONJ', 'DET', 'PRON', 'ADJ', 'NOUN', 'PROPN', 'SYM', 'PUNCT']
# ENT_SET = ['DATE', 'PERSON', 'QUANTITY', 'LOC', 'ORG', 'FAC', 'ORDINAL', 'WORK_OF_ART', 
#            'EVENT', 'PRODUCT', 'LAW', 'MONEY', 'NORP', 'TIME', 'CARDINAL', 'LANGUAGE', 'GPE', 'PERCENT']
# pipline = spacy.load('en_core_web_md')

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
            data[domain] = DataProcessor.read_file("../data/snips/" + domain + '/' + domain + '.txt')
        data['atis'] = DataProcessor.read_file("../data/atis/atis.txt")
        return data
    
    @staticmethod
    def convert_rowdata_to_mrc(all_data, all_slots):
        '''
        label_knowledges: list[str], 长度为39, 所有slot的label_knowledge
        all_slots: list[str], 长度为39, 和label_knowledge是一一对应的关系
        '''
        samples = []
        # slot2query = {}
        # label_knowleges = []
        # if query_type == "desp":
        #     for slot, desp in slot2desp.items():
        #         # slot2query[slot] = "what is the {}".format(desp)
        #         slot2query[slot] = desp
        #         # logger.info(desp)
        #     # logger.info('Using queries from description.')
        # assert len(slot2query) == 39
        context_id = 0
        # for _, (label, query) in enumerate(slot2query.items()):
        #     # if label not in domain2slots[dm_name]:
        #     #     continue
        #     logger.info(query)
        #     label_knowleges.append(query)
        for line in tqdm(all_data):
            src, labels = line.strip().split("\t")
            # src = src.strip()
            # src = re.sub(r'\.+', '', src)
            # src = re.sub(r'\'', '', src)
            # src = re.sub(r':', '', src)
            # src = re.sub(r'\'', '', src)
            # src = re.sub(r'\"', '', src)
            # src = re.sub(r'’', '', src)
            # src = re.sub(r'-', '', src)
            # src = re.sub(r'&', '', src)
            # src = re.sub(r'/', '', src)
            # src = re.sub(r'\s+', ' ', src)
            # doc = pipline(src)
            # if len(doc) != len(src.split()) or len(src.split()) != len(labels.split()):
            #     continue
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
            all_start_positions = []
            all_end_positions = []
            all_tags = []

            # for _, (label, _) in enumerate(slot2query.items()):
            for label in all_slots:
                # if label not in domain2slots[dm_name]:
                #     continue
                start_position = [tag['begin'] for tag in tags if tag['tag'] == label]
                end_position = [tag['end'] - 1 for tag in tags if tag['tag'] == label]
                all_start_positions.append(start_position)
                all_end_positions.append(end_position)
                all_tags.append(label)  # list[str] len=num_labels
            samples.append(
                {
                    'context_id': context_id,
                    'context_src': src,
                    'label_src': labels,
                    'all_start_positions': all_start_positions,  # [num_label] list
                    'all_end_positions': all_end_positions,  # [num_label] str
                    'all_tags': all_tags  # [num_label] str
                }
            )
            context_id += 1
        return samples


class LabelEnhancedDataset(Dataset):
    def __init__(self, all_data, tokenizer, label_knowleges, context_max_len=64, label_max_len=16):
        super().__init__()
        self.all_data = all_data
        self.tokenizer = tokenizer
        self.context_max_len = context_max_len
        self.label_max_len = label_max_len
        self.label_knowleges = label_knowleges
        # self.domain_name = domain_name
        self.num_labels = len(label_knowleges)
        
        self.label_knowlege_token = []
        self.label_knowlege_attention_mask = []
        self.label_knowlege_token_type_ids = []
        self.label_knowlege_pos_token = []
        self.label_knowlege_ent_token = []
        for i in range(self.num_labels):
            tokens = []
            attention_mask = []
            knowledge_str = self.label_knowleges[i]
            knowlege_words = []
            # pos_tokens = []
            # ents_tmp = []
            # doc = pipline(knowledge_str)
            doc = knowledge_str.strip().split()
            for token in doc:
                # knowlege_words.append(token.text)
                knowlege_words.append(token)
                # pos_tokens.append(POS_SET.index(token.pos_) + 1)
            # ent_tokens = [0] * len(pos_tokens)
            # for ent in doc.ents:
            #     ents_tmp.append((ENT_SET.index(ent.label_) + 1, ent.start, ent.end))
            # for ent_label, start, end in ents_tmp:
            #     for i in range(start, end):
            #         ent_tokens[i] = ent_label
            
            # pos_subword_token = []
            # ent_subword_token = []
            for idx, word in enumerate(knowlege_words):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                attention_mask.extend([1] * len(token))
                # pos_subword_token = pos_subword_token + [pos_tokens[idx]] * len(token)
                # ent_subword_token = ent_subword_token + [ent_tokens[idx]] * len(token)
            # add cls
            tokens = [self.tokenizer.cls_token] + tokens
            # pos_subword_token = [0] + pos_subword_token
            # ent_subword_token = [0] + ent_subword_token
            attention_mask = [1] + attention_mask
            # truncate
            if len(tokens) > self.label_max_len - 1:
                tokens = tokens[:self.label_max_len - 1]
                attention_mask = attention_mask[:self.label_max_len - 1]
                # pos_subword_token = pos_subword_token[:self.label_max_len - 1]
                # ent_subword_token = ent_subword_token[:self.label_max_len - 1]
            # add sep
            tokens = tokens + [self.tokenizer.sep_token]
            attention_mask = attention_mask + [1]
            # pos_subword_token = pos_subword_token + [0]
            # ent_subword_token = ent_subword_token + [0]
            # pad
            length = len(tokens)
            for _ in range(self.label_max_len - length):
                tokens = tokens + [self.tokenizer.pad_token]
                attention_mask = attention_mask + [0]
                # pos_subword_token = pos_subword_token + [0]
                # ent_subword_token = ent_subword_token + [0]
            token_type_ids = [0] * self.label_max_len
            # assert len(pos_subword_token) == len(ent_subword_token) == self.label_max_len
            self.label_knowlege_token.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.label_knowlege_attention_mask.append(attention_mask)
            self.label_knowlege_token_type_ids.append(token_type_ids)
            # self.label_knowlege_pos_token.append(pos_subword_token)
            # self.label_knowlege_ent_token.append(ent_subword_token)
        
        # self.label_knowlege_token = torch.tensor(self.label_knowlege_token, dtype=torch.long)
        # self.label_knowlege_attention_mask = torch.tensor(self.label_knowlege_attention_mask, dtype=torch.long)
        # self.label_knowlege_token_type_ids = torch.tensor(self.label_knowlege_token_type_ids, dtype=torch.long)
        self.label_knowlege_token = torch.tensor(self.label_knowlege_token, dtype=torch.long)  # [num_labels, L2]
        self.label_knowlege_attention_mask = torch.tensor(self.label_knowlege_attention_mask, dtype=torch.long)
        self.label_knowlege_token_type_ids = torch.tensor(self.label_knowlege_token_type_ids, dtype=torch.long)
        # self.label_knowlege_pos_token = torch.tensor(self.label_knowlege_pos_token, dtype=torch.long)
        # self.label_knowlege_ent_token = torch.tensor(self.label_knowlege_ent_token, dtype=torch.long)


    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        data = self.all_data[index]
        context = []  # str
        label_src = data['label_src']  # str
        all_start_positions = data['all_start_positions']  # [num_labels] list
        all_end_positions = data['all_end_positions']  # [num_labels] list
        all_tags = data['all_tags']  # [num_labels] str


        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token

        context_tmp = data['context_src']
        # doc = pipline(context_tmp)
        # pos_tokens = []
        # for token in doc:
        #     context.append(token.text)
        #     pos_tokens.append(POS_SET.index(token.pos_) + 1)
        # ent_tokens = [0] * len(pos_tokens)
        # ents_tmp = []
        # for ent in doc.ents:
        #     ents_tmp.append((ENT_SET.index(ent.label_) + 1, ent.start, ent.end))
        # for ent_label, start, end in ents_tmp:
        #     for i in range(start, end):
        #         ent_tokens[i] = ent_label

        # assert len(context_tmp.split()) == len(context), f'{context}, {context_tmp}'

        # pos_subword_tokens = []
        # ent_subword_tokens = []

        context = context_tmp.strip().split()
        context_token = []
        start_token_mask = []
        end_token_mask = []
        context_token_to_origin_index = []
        start_labels = []
        end_labels = []

        for idx, word in enumerate(context):
            token = self.tokenizer.tokenize(word)
            context_token.extend(token)
            for j in range(len(token)):
                # pos_subword_tokens.append(pos_tokens[idx])
                # ent_subword_tokens.append(ent_tokens[idx])
                context_token_to_origin_index.append(idx)
                if j == 0:
                    start_token_mask.append(1)
                else:
                    start_token_mask.append(0)
                if j == len(token) - 1:
                    end_token_mask.append(1)
                else:
                    end_token_mask.append(0)

        for i in range(len(all_start_positions)):
            start_position = all_start_positions[i]
            start_label = []
            for idx, word in enumerate(context):
                tokens = self.tokenizer.tokenize(word)
                for j, _ in enumerate(tokens):
                    if idx in start_position and j == 0:
                        start_label.append(1)
                    else:
                        start_label.append(0)
            start_labels.append(start_label)  # [num_labels, L1]
        for i in range(len(all_end_positions)):
            end_position = all_end_positions[i]
            end_label = []
            for idx, word in enumerate(context):
                tokens = self.tokenizer.tokenize(word)
                for j, _ in enumerate(tokens):
                    if idx in end_position and j == len(tokens) - 1:
                        end_label.append(1)
                    else:
                        end_label.append(0)
            end_labels.append(end_label)  # [num_labels, L1]

        assert len(context_token) == len(start_labels[0]) == len(end_labels[0])
        assert len(all_end_positions) == len(all_start_positions) == self.num_labels
        assert len(start_labels) == len(end_labels) == self.num_labels

        # add cls
        context_token = [cls_token] + context_token
        start_token_mask = [0] + start_token_mask
        end_token_mask = [0] + end_token_mask
        context_token_to_origin_index = [0] + context_token_to_origin_index
        for i in range(len(start_labels)):
            start_labels[i] = [0] + start_labels[i]
            end_labels[i] = [0] + end_labels[i]
        # pos_subword_tokens = [0] + pos_subword_tokens
        # ent_subword_tokens = [0] + ent_subword_tokens

        # truncate
        if len(context_token) > self.context_max_len - 1:
            context_token = context_token[:self.context_max_len - 1]
            start_token_mask = start_token_mask[:self.context_max_len - 1]
            end_token_mask = end_token_mask[:self.context_max_len - 1]
            context_token_to_origin_index = context_token_to_origin_index[:self.context_max_len - 1]
            for i in range(self.num_labels):
                start_labels[i] = start_labels[i][:self.context_max_len - 1]
                end_labels[i] = end_labels[i][:self.context_max_len - 1]
            # pos_subword_tokens = pos_subword_tokens[:self.context_max_len - 1]
            # ent_subword_tokens = ent_subword_tokens[:self.context_max_len - 1]

        # add sep
        context_token = context_token + [sep_token]
        start_token_mask = start_token_mask + [0]
        end_token_mask = end_token_mask + [0]
        # pos_subword_tokens = pos_subword_tokens + [0]
        # ent_subword_tokens = ent_subword_tokens + [0]
        context_token_to_origin_index = context_token_to_origin_index + [0]
        for i in range(self.num_labels):
            start_labels[i] = start_labels[i] + [0]
            end_labels[i] = end_labels[i] + [0]
        
        attention_mask = [1] * len(context_token)
        token_type_ids = [0] * len(context_token)

        # add pad
        length = len(context_token)
        for _ in range(self.context_max_len - length):
            context_token = context_token + [pad_token]
            attention_mask = attention_mask + [0]
            token_type_ids = token_type_ids + [0]
            start_token_mask = start_token_mask + [0]
            end_token_mask = end_token_mask + [0]
            context_token_to_origin_index = context_token_to_origin_index + [0]
            for i in range(self.num_labels):
                start_labels[i] = start_labels[i] + [0]
                end_labels[i] = end_labels[i] + [0]
            # pos_subword_tokens = pos_subword_tokens + [0]
            # ent_subword_tokens = ent_subword_tokens + [0]

        input_ids = self.tokenizer.convert_tokens_to_ids(context_token)



        return{
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_labels': torch.tensor(start_labels, dtype=torch.long),  # [num_labels, context_max_len]
            'end_labels': torch.tensor(end_labels, dtype=torch.long),
            'start_token_mask': torch.tensor(start_token_mask, dtype=torch.long),  # [context_max_len]
            'end_token_mask': torch.tensor(end_token_mask, dtype=torch.long),  # [context_max_len]
            'label_src': label_src,  # str
            'tags': all_tags,  # [num_labels] str
            'context_token_to_origin_index': context_token_to_origin_index,  # [list] int
            'context_src': data['context_src'],  # str
            'context_id': data['context_id'],  # int
            'label_knowledges_input_ids': self.label_knowlege_token,  # tensor [num_labels, L2]
            'label_knowledges_attention_mask': self.label_knowlege_attention_mask,  # tensor [num_labels, L2]
            'label_knowledges_token_type_ids': self.label_knowlege_token_type_ids,  # tensor [num_labels, L2]
            'num_labels': self.num_labels  # int
            # 'pos_token': torch.tensor(pos_subword_tokens, dtype=torch.long),
            # 'ent_token': torch.tensor(ent_subword_tokens, dtype=torch.long)
        }


def collate_fn(batch):  # batch是字典的列表
    output = {}
    output['input_ids'] = torch.stack([x['input_ids'] for x in batch], dim=0)
    output['attention_mask'] = torch.stack([x['attention_mask'] for x in batch], dim=0)
    output['token_type_ids'] = torch.stack([x['token_type_ids'] for x in batch], dim=0)
    output['start_labels'] = torch.stack([x['start_labels'] for x in batch], dim=0)
    output['end_labels'] = torch.stack([x['end_labels'] for x in batch], dim=0)
    output['start_token_mask'] = torch.stack([x['start_token_mask'] for x in batch], dim=0)
    output['end_token_mask'] = torch.stack([x['end_token_mask'] for x in batch], dim=0)
    # output['pos_token'] = torch.stack([x['pos_token'] for x in batch], dim=0)
    # output['ent_token'] = torch.stack([x['ent_token'] for x in batch], dim=0)
    output['label_src'] = []
    output['label_src'].extend([x['label_src'] for x in batch])
    output['tags'] = []
    output['tags'].extend([x['tags'] for x in batch])
    output['context_token_to_origin_index'] = []
    output['context_token_to_origin_index'].extend([x['context_token_to_origin_index'] for x in batch])
    output['context_src'] = []
    output['context_src'].extend([x['context_src'] for x in batch])
    output['context_id'] = []
    output['context_id'].extend([x['context_id'] for x in batch])
    output['label_knowledges_input_ids'] = batch[0]['label_knowledges_input_ids']  # tensor, [num_labels, L1]
    output['label_knowledges_attention_mask'] = batch[0]['label_knowledges_attention_mask']  # tensor, [num_labels, L1]
    output['label_knowledges_token_type_ids'] = batch[0]['label_knowledges_token_type_ids']  # tensor, [num_labels, L1]
    output['num_labels'] = batch[0]['num_labels']  # int
    return output


def get_dataset(tgt_domain, n_samples, tokenizer, context_max_len=64, label_max_len=16, query_type="desp"):
    label_knowledges = []
    all_slots = []
    if query_type == "desp":
        for slot, desp in slot2desp.items():
            label_knowledges.append(desp)
            all_slots.append(slot)

    assert len(label_knowledges) == 39
    all_row_data = DataProcessor.get_all_data()
    all_row_valid_data = []
    all_row_test_data = []
    all_row_train_data = []
    for domain_name, data in all_row_data.items():
        if domain_name == tgt_domain:
            all_row_train_data = all_row_train_data + data[: n_samples]
            all_row_valid_data = data[n_samples: 500]
            all_row_test_data = data[500:]
            continue
        if domain_name != 'atis':
            all_row_train_data = all_row_train_data + data
    

    all_train_data = DataProcessor.convert_rowdata_to_mrc(all_row_train_data, all_slots)
    all_valid_data = DataProcessor.convert_rowdata_to_mrc(all_row_valid_data, all_slots)
    all_test_data = DataProcessor.convert_rowdata_to_mrc(all_row_test_data, all_slots)

    train_dataset = LabelEnhancedDataset(all_train_data, tokenizer, label_knowledges, context_max_len=context_max_len,
                                         label_max_len=label_max_len)
    valid_dataset = LabelEnhancedDataset(all_valid_data, tokenizer, label_knowledges, context_max_len=context_max_len,
                                         label_max_len=label_max_len)
    test_dataset = LabelEnhancedDataset(all_test_data, tokenizer, label_knowledges, context_max_len=context_max_len,
                                         label_max_len=label_max_len)

    # train_datasets = []
    # for dm_name, dm_row_data in all_row_data.items():  # dm_data是原始字符串的列表
    #     if dm_name != tgt_domain and dm_name != 'atis':  # atis只做测试用
    #         domain_data = DataProcessor.convert_rowdata_to_mrc(dm_row_data, all_slots=all_slots)
    #         dataset = LabelEnhancedDataset(domain_data, tokenizer, label_knowleges, dm_name, context_max_len=context_max_len, label_max_len=label_max_len)
    #         train_datasets.append(dataset)

    # if n_samples != 0:
    #     domain_data = DataProcessor.convert_rowdata_to_mrc(
    #         all_row_data[tgt_domain][:n_samples],
    #         all_slots=all_slots
    #     )
    #     dataset = LabelEnhancedDataset(
    #         domain_data, 
    #         tokenizer, 
    #         label_knowleges, 
    #         tgt_domain, 
    #         context_max_len=context_max_len, 
    #         label_max_len=label_max_len
    #     )
    #     train_datasets.append(dataset)
    

    # valid_label_knowleges, valid_domain_data = DataProcessor.convert_rowdata_to_mrc(
    #     all_row_data[tgt_domain][n_samples:500],
    #     all_slots=all_slots
    # )
    # valid_dataset = LabelEnhancedDataset(
    #     valid_domain_data,
    #     tokenizer,
    #     valid_label_knowleges,
    #     tgt_domain,
    #     context_max_len=context_max_len,
    #     label_max_len=label_max_len 
    # )

    # test_label_knowleges, test_domain_data = DataProcessor.convert_rowdata_to_mrc(
    #     all_row_data[tgt_domain][500:],
    #     all_slots=all_slots
    # )
    # test_dataset = LabelEnhancedDataset(
    #     test_domain_data,
    #     tokenizer,
    #     test_label_knowleges,
    #     tgt_domain,
    #     context_max_len=context_max_len,
    #     label_max_len=label_max_len
    # )

    logger.info('********** Data information : **********')
    logger.info(f'The target domain is {tgt_domain}')
    logger.info(f'Train dataset length = {len(train_dataset)}')
    # logger.info('********** Valid data information : **********')
    logger.info(f'Valid dataset length = {len(valid_dataset)}')
    # logger.info('********** Test data information : **********')
    logger.info(f'Test dataset length = {len(test_dataset)}')
    logger.info('********** Data information : **********')
    return train_dataset, valid_dataset, test_dataset



if __name__ == '__main__':
    # from paddlenlp.transformers import BertTokenizer
    # from paddle.io import DataLoader
    # paddle.device.set_device('cpu')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # tgt_domain = 'BookRestaurant'
    # n_samples = 0
    # train_datasets, valid_dataset, test_dataset = get_dataset(
    #     tgt_domain,
    #     n_samples,
    #     tokenizer
    # )
    # valid_data = DataLoader(valid_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # for i in valid_data:
    #     print(i)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset, valid_dataset, test_dataset = get_dataset('AddToPlaylist', 0, tokenizer)
    pass


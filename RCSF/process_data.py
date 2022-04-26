from dataset import DataProcessor
import spacy
from spacy.tokens import Doc
from utils import domain_set
import json
from tqdm import tqdm
import os

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True] * len(words)
            
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'lemmatizer'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def main():
    all_data = DataProcessor.get_all_data()
    for tgt_domain in domain_set:
        train_data = []
        for dm_name, dm_data in all_data.items():  # dm_data是原始字符串的列表
            if dm_name != tgt_domain and dm_name != 'atis':  # atis只做测试用
                train_data.extend(DataProcessor.convert_rowdata_to_mrc(dm_data, dm_name, query_type='desp'))
    
    
        valid_data = DataProcessor.convert_rowdata_to_mrc(all_data[tgt_domain][: 500], tgt_domain, query_type='desp')
        test_data = DataProcessor.convert_rowdata_to_mrc(all_data[tgt_domain][500:], tgt_domain, query_type='desp')

        processed_train_data = []
        processed_valid_data = []
        processed_test_data = []

        for data in tqdm(train_data, ncols=100):
            query = data['query_src']
            context = data['context_src']

            context_doc = nlp(context.strip())
            query_doc = nlp(query.strip())
            context = [token.text for token in context_doc]  # spacy分词后的列表
            query = [token.text for token in query_doc]  # spacy分词后的列表
            query_span = [(token.left_edge.i, token.right_edge.i + 1) for token in query_doc]
            context_span = [(token.left_edge.i, token.right_edge.i + 1) for token in context_doc]
            query_head = [token.head.text for token in query_doc]
            context_head = [token.head.text for token in context_doc]
            query_dep = [token.dep_ for token in query_doc]
            context_dep = [token.dep_ for token in context_doc]

            processed_train_data.append(
                {
                    'context_id': data['context_id'],
                    'tag_id': data['tag_id'],
                    'context_src': data['context_src'],
                    'label_src': data['label_src'],
                    'query_src': data['query_src'],
                    'tag': data['tag'],
                    'start_positions': data['start_positions'],
                    'end_positions': data['end_positions'],
                    'query_span': query_span,
                    'query_head': query_head,
                    'context_span': context_span,
                    'context_head': context_head,
                    'query_dep': query_dep,
                    'context_dep': context_dep
                }
            )

        for data in tqdm(valid_data, ncols=100):
            query = data['query_src']
            context = data['context_src']

            context_doc = nlp(context.strip())
            query_doc = nlp(query.strip())
            context = [token.text for token in context_doc]  # spacy分词后的列表
            query = [token.text for token in query_doc]  # spacy分词后的列表
            query_span = [(token.left_edge.i, token.right_edge.i + 1) for token in query_doc]
            context_span = [(token.left_edge.i, token.right_edge.i + 1) for token in context_doc]
            query_head = [token.head.text for token in query_doc]
            context_head = [token.head.text for token in context_doc]

            query_dep = [token.dep_ for token in query_doc]
            context_dep = [token.dep_ for token in context_doc]

            processed_valid_data.append(
                {
                    'context_id': data['context_id'],
                    'tag_id': data['tag_id'],
                    'context_src': data['context_src'],
                    'label_src': data['label_src'],
                    'query_src': data['query_src'],
                    'tag': data['tag'],
                    'start_positions': data['start_positions'],
                    'end_positions': data['end_positions'],
                    'query_span': query_span,
                    'query_head': query_head,
                    'context_span': context_span,
                    'context_head': context_head,
                    'query_dep': query_dep,
                    'context_dep': context_dep
                }
            )

        for data in tqdm(test_data, ncols=100):
            query = data['query_src']
            context = data['context_src']

            context_doc = nlp(context.strip())
            query_doc = nlp(query.strip())
            context = [token.text for token in context_doc]  # spacy分词后的列表
            query = [token.text for token in query_doc]  # spacy分词后的列表
            query_span = [(token.left_edge.i, token.right_edge.i + 1) for token in query_doc]
            context_span = [(token.left_edge.i, token.right_edge.i + 1) for token in context_doc]
            query_head = [token.head.text for token in query_doc]
            context_head = [token.head.text for token in context_doc]

            query_dep = [token.dep_ for token in query_doc]
            context_dep = [token.dep_ for token in context_doc]

            processed_test_data.append(
                {
                    'context_id': data['context_id'],
                    'tag_id': data['tag_id'],
                    'context_src': data['context_src'],
                    'label_src': data['label_src'],
                    'query_src': data['query_src'],
                    'tag': data['tag'],
                    'start_positions': data['start_positions'],
                    'end_positions': data['end_positions'],
                    'query_span': query_span,
                    'query_head': query_head,
                    'context_span': context_span,
                    'context_head': context_head,
                    'query_dep': query_dep,
                    'context_dep': context_dep
                }
            )

        if not os.path.exists(f'../data/snips_processed/{tgt_domain}'):
            os.makedirs(f'../data/snips_processed/{tgt_domain}')
        with open(f'../data/snips_processed/{tgt_domain}/train.json', 'w', encoding='utf-8') as f:
            json.dump(processed_train_data, f, indent=4)
        with open(f'../data/snips_processed/{tgt_domain}/valid.json', 'w', encoding='utf-8') as f:
            json.dump(processed_valid_data, f, indent=4)
        with open(f'../data/snips_processed/{tgt_domain}/test.json', 'w', encoding='utf-8') as f:
            json.dump(processed_test_data, f, indent=4)
        
        print(f'Target domain = {tgt_domain}, train_data length = {len(train_data)}, valid_data length = {len(valid_data)}, test_data length = {len(test_data)}')


if __name__ == '__main__':
    main()
    # doc = nlp('what is the playlist')
    # tokens = [token.text for token in doc]
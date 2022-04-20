import spacy
import os
import re
from tqdm import tqdm


# domain = 'AddToPlaylist'
# data_file = os.path.join('data/snips', domain, domain + '.txt')

# with open(data_file, 'r', encoding='utf-8') as f:
#     all_data = f.readlines()

# lines = [data.split('\t')[0].lower().strip() for data in all_data]
# labels = [data.split('\t')[1].lower().strip() for data in all_data]

# lines = lines[:4]
# batch_size = 3
# for i in range(0, len(lines), batch_size):
#     lines_tmp = lines[i: i + batch_size]
#     docs = pipline.pipe(lines_tmp)
#     for doc in docs:
#         for token in doc:
#             print(token.pos_)
#         # for ent in doc.ents:
#         #     print(ent.text, ent.label_, ent.start_char, ent.start, ent.end_char, ent.end)
#         # for chunk in doc.noun_chunks:
#         #     print(chunk.text, chunk.start, chunk.end)

pipline = spacy.load('en_core_web_md')

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

pos_set = set()
ent_set = set()
batch_size = 16
cnt = 0

for domain in domain_set:
    data_file = os.path.join('data/snips', domain, domain + '.txt')
    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = f.readlines()
    lines = [data.split('\t')[0].lower().strip() for data in all_data]
    labels = [data.split('\t')[1].lower().strip() for data in all_data]
    for line, label in tqdm(zip(lines, labels), desc=f'{domain}'):
        line = line.strip()
        line = re.sub(r'\.+', '', line)
        line = re.sub(r'\'', '', line)
        line = re.sub(r':', '', line)
        line = re.sub(r'\'', '', line)
        line = re.sub(r'\"', '', line)
        line = re.sub(r'’', '', line)
        line = re.sub(r'-', '', line)
        line = re.sub(r'&', '', line)
        line = re.sub(r'/', '', line)
        line = re.sub(r'\s+', ' ', line)
        doc = pipline(line)
        if len(doc) != len(line.split()) or len(line.split()) != len(label.split()):
            print(line)
            print(len(line.split()))
            print(len(doc))
            print(len(label.split()))
            cnt += 1

print(cnt)
        



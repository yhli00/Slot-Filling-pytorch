import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True] * len(words)
            
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'lemmatizer'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
# text = 'Autonomous cars shift insurance liability toward manufacturers'
text = 'add a tom thacker tune to my rock classics'
doc = nlp(text)
# for token in doc:
#     print(token.text, token.dep_, token.head.text, [child for child in token.children])
for token in doc:
    print(token.text, token.head.text, token.left_edge.i, token.right_edge.i)

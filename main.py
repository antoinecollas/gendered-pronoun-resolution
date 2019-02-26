import csv, os,sys, torch, logging
from utils import DataLoader, compute_word_pos
from pytorch_pretrained_bert import BertTokenizer, BertModel

FOLDER = 'gap-coreference'
TRAINING_PATH = os.path.join(FOLDER, 'gap-development.tsv')
VAL_PATH = os.path.join(FOLDER, 'gap-validation.tsv')

BERT_MODEL = 'bert-large-cased'
BATCH_SIZE = 32

data_training = DataLoader(TRAINING_PATH, BATCH_SIZE, shuffle=True)

print('Columns:', data_training.get_col_names())
print('Nb training examples:', len(data_training))

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

for X, Y in data_training:
    # data pre processing
    data_processed = list()
    for row in X.itertuples(index=False):
        tokenized_text = tokenizer.tokenize(row.Text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        pos_pronoun = compute_word_pos(row.Text, tokenized_text, row.Pronoun, row._2)
        pos_A = compute_word_pos(row.Text, tokenized_text, row.A, row._4)
        pos_B = compute_word_pos(row.Text, tokenized_text, row.B, row._6)
        data_processed.append([indexed_tokens, pos_pronoun, pos_A, pos_B])
    
    print(data_processed)
    sys.exit(0)    
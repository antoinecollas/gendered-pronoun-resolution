import csv, os,sys, torch, logging
from git import Repo
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos
from pytorch_pretrained_bert import BertTokenizer, BertModel

DEBUG = True

FOLDER = 'gap-coreference'
if not os.path.exists(FOLDER):
    Repo.clone_from('https://github.com/google-research-datasets/gap-coreference', FOLDER)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', DEVICE)

TRAINING_PATH = os.path.join(FOLDER, 'gap-development.tsv')
VAL_PATH = os.path.join(FOLDER, 'gap-validation.tsv')

if DEBUG:
    BERT_MODEL = 'bert-base-cased'
    BATCH_SIZE = 2
else:
    BERT_MODEL = 'bert-large-cased'
    BATCH_SIZE = 32

data_training = DataLoader(TRAINING_PATH, BATCH_SIZE, shuffle=True)

print('Nb training examples:', len(data_training))

# load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
pad_token = tokenizer.tokenize("[PAD]")
PAD_ID = tokenizer.convert_tokens_to_ids(pad_token)[0]

# load pretrained model
model = BertModel.from_pretrained(BERT_MODEL)
model.eval()
model.to(DEVICE)


for X, Y in data_training:
    # data pre processing
    tokens, pos = list(), list()
    for row in X.itertuples(index=False):
        tokenized_text = tokenizer.tokenize(row.Text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        pos_pronoun = compute_word_pos(row.Text, tokenized_text, row.Pronoun, row._2)
        pos_A = compute_word_pos(row.Text, tokenized_text, row.A, row._4)
        pos_B = compute_word_pos(row.Text, tokenized_text, row.B, row._6)
        pos.append([pos_pronoun, pos_A, pos_B])
    pos = torch.Tensor(pos).long()

    tokens = pad(tokens, PAD_ID)
    tokens = torch.tensor(tokens).to(DEVICE)
    attention_mask = torch.ones(tokens.shape).to(DEVICE)
    attention_mask[tokens==PAD_ID] = 0
    
    with torch.no_grad():
        encoded_layers, _ = model(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]

    vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
    res = torch.cat(vect_wordpiece, dim=1)
    print(res.shape)

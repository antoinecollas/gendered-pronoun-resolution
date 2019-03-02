import csv, os,sys, torch, logging
from tqdm import tqdm
from git import Repo
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, print_tensorboard
from pytorch_pretrained_bert import BertTokenizer, BertModel
from neural_nets import MLP
from tensorboardX import SummaryWriter
import numpy as np

DEBUG = False
writer = SummaryWriter()

FOLDER = 'gap-coreference'
if not os.path.exists(FOLDER):
    Repo.clone_from('https://github.com/google-research-datasets/gap-coreference', FOLDER)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', DEVICE)

TRAINING_PATH = os.path.join(FOLDER, 'gap-development.tsv')
VAL_PATH = os.path.join(FOLDER, 'gap-validation.tsv')

NB_EPOCHS = 1000

if DEBUG:
    BERT_MODEL = 'bert-base-cased'
    classifier = MLP(3*768, 3) # output: nothing, A, B
    BATCH_SIZE = 2
else:
    BERT_MODEL = 'bert-large-cased'
    classifier = MLP(3*1024, 3)
    BATCH_SIZE = 32

classifier.train().to(DEVICE)
print('number of parameters:', torch.nn.utils.parameters_to_vector(classifier.parameters()).shape[0])

# load pre-trained Bert tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
pad_token = tokenizer.tokenize("[PAD]")
PAD_ID = tokenizer.convert_tokens_to_ids(pad_token)[0]

# load pretrained Bert
Bert = BertModel.from_pretrained(BERT_MODEL)
Bert.eval()
Bert.to(DEVICE)

loss = torch.nn.CrossEntropyLoss()
loss_values = list()
optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9)

for epoch in tqdm(range(NB_EPOCHS)):
    data_training = DataLoader(TRAINING_PATH, BATCH_SIZE, shuffle=True, debug=DEBUG)
    if epoch ==0:
        print('len(data_training):', len(data_training))

    for X, Y in data_training:
        temp = torch.Tensor(Y.values.astype(int))
        Y = torch.zeros((Y.shape[0], Y.shape[1]+1)) 
        Y[:,1:3] = temp
        Y[:,0] = 1 - Y.sum(dim=1)
        Y = torch.argmax(Y, dim=1).to(DEVICE)

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
            encoded_layers, _ = Bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]

        vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
        features = torch.cat(vect_wordpiece, dim=1)

        output = classifier(features)
        optimizer.zero_grad()
        output = loss(output, Y)
        loss_values.append(output.item())
        output.backward()
        optimizer.step()

    scalars = {
        'training/loss': np.mean(loss_values),
        'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(classifier.parameters()), p=2),   
    }
    print_tensorboard(writer, scalars, epoch)
    

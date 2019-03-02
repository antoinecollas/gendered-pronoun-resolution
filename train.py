import csv, os,sys, torch, logging
from tqdm import tqdm
from git import Repo
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard
from pytorch_pretrained_bert import BertTokenizer, BertModel
from neural_nets import MLP
from tensorboardX import SummaryWriter
import numpy as np

DEBUG = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', DEVICE)
writer = SummaryWriter()

FOLDER_DATA = 'gap-coreference'
if not os.path.exists(FOLDER_DATA):
    Repo.clone_from('https://github.com/google-research-datasets/gap-coreference', FOLDER_DATA)
TRAINING_PATH = os.path.join(FOLDER_DATA, 'gap-development.tsv')
VAL_PATH = os.path.join(FOLDER_DATA, 'gap-validation.tsv')

NB_EPOCHS = 1000

if DEBUG:
    BERT_MODEL = 'bert-base-cased'
    classifier = MLP(3*768, 3) # output: A, B, neither
    BATCH_SIZE = 2
    EVALUATION_FREQUENCY = 2
else:
    BERT_MODEL = 'bert-large-cased'
    classifier = MLP(3*1024, 3)
    BATCH_SIZE = 32
    EVALUATION_FREQUENCY = 5

classifier.to(DEVICE)
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
loss_values, loss_values_eval = list(), list()
optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,100], gamma=0.1)

for epoch in tqdm(range(NB_EPOCHS)):
    scheduler.step()
    data_training = DataLoader(TRAINING_PATH, BATCH_SIZE, shuffle=True, debug=DEBUG)

    for X, Y in data_training:
        tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, DEVICE, PAD_ID)
        
        with torch.no_grad():
            encoded_layers, _ = Bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]
        vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
        features = torch.cat(vect_wordpiece, dim=1)
        
        classifier.train()
        output = classifier(features)
        
        optimizer.zero_grad()
        output = loss(output, Y)
        loss_values.append(output.item())
        output.backward()
        optimizer.step()

    if epoch%EVALUATION_FREQUENCY == 0:
        data_eval = DataLoader(VAL_PATH, BATCH_SIZE, shuffle=True, debug=DEBUG)
        for X, Y in data_eval:
            tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, DEVICE, PAD_ID)
            
            with torch.no_grad():
                encoded_layers, _ = Bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]
            vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
            features = torch.cat(vect_wordpiece, dim=1)
            
            classifier.eval()
            with torch.no_grad():
                output = classifier(features)
            output = loss(output, Y)
            loss_values_eval.append(output.item())

        scalars = {
            'training/loss': np.mean(loss_values),
            'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(classifier.parameters()), p=2),
            'eval/loss'  : np.mean(loss_values_eval)
        }
        print_tensorboard(writer, scalars, epoch)
        loss_values, loss_values_eval = list(), list()
        torch.save(classifier.state_dict(), 'weights_classifier')
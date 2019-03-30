import torch
import torch.nn as nn
from torch.nn.functional import softmax
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import preprocess_data, get_vect_from_pos
import pandas as pd

class Pooling(nn.Module):
    def __init__(self, in_features, d_proj=256):
        super(Pooling, self).__init__()
        self.proj_pronoun = nn.Linear(in_features, d_proj)
        self.proj_other = nn.Linear(in_features, d_proj)
        self.att_pronoun = nn.Linear(d_proj, 1, bias=False)
        self.att_other = nn.Linear(d_proj, 1, bias=False)

    def forward(self, x):
        pronoun, A, B = x
        for i in range(len(pronoun)):
            pronoun[i] = self.proj_pronoun(pronoun[i])
            A[i] = self.proj_other(A[i])
            B[i] = self.proj_other(B[i])
        
        weights_pronoun, weights_A, weights_B = list(), list(), list()
        for i in range(len(pronoun)):
            weights_pronoun.append(softmax(self.att_pronoun(pronoun[i]).reshape(-1), dim=0))
            weights_A.append(softmax(self.att_other(A[i]).reshape(-1), dim=0))
            weights_B.append(softmax(self.att_other(B[i]).reshape(-1), dim=0))
        
        for i in range(len(pronoun)):
            pronoun[i] = torch.sum(pronoun[i]*weights_pronoun[i].unsqueeze(1), dim=0)
            A[i] = torch.sum(A[i]*weights_A[i].unsqueeze(1), dim=0)
            B[i] = torch.sum(B[i]*weights_B[i].unsqueeze(1), dim=0)

        pronoun = torch.stack(pronoun)
        A = torch.stack(A)
        B = torch.stack(B)
        
        return [pronoun, A, B]

class MLP(nn.Module):
    def __init__(self, d_proj, dropout=0.2, d_hid = 512):
        super(MLP, self).__init__()
        in_features = 3*d_proj
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_features), 
            nn.Dropout(dropout),
            nn.Linear(in_features, d_hid),
            nn.ReLU(), 
            nn.BatchNorm1d(d_hid), 
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.BatchNorm1d(d_hid),
            nn.Dropout(dropout),
            nn.Linear(d_hid, 4), #none, A, B, A&B
        )

    def forward(self, features):
        return self.mlp(features)

class Model():
    def __init__(self, cfg):
        if cfg.DEBUG:
            BERT_MODEL = 'bert-base-uncased'
            self.pooling = Pooling(768, cfg.D_PROJ).to(cfg.DEVICE)
        else:
            BERT_MODEL = 'bert-large-uncased'
            self.pooling = Pooling(1024, cfg.D_PROJ).to(cfg.DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        pad_token = self.tokenizer.tokenize("[PAD]")
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(pad_token)[0]
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.bert.to(cfg.DEVICE).eval()
        self.mlp = MLP(cfg.D_PROJ)
        self.mlp.to(cfg.DEVICE)
        self.DEVICE = cfg.DEVICE
        self.PATH_WEIGHTS_POOLING_ONTONOTES = cfg.PATH_WEIGHTS_POOLING_ONTONOTES
        self.PATH_WEIGHTS_CLASSIFIER_ONTONOTES = cfg.PATH_WEIGHTS_CLASSIFIER_ONTONOTES
        self.PATH_WEIGHTS_POOLING = cfg.PATH_WEIGHTS_POOLING
        self.PATH_WEIGHTS_CLASSIFIER = cfg.PATH_WEIGHTS_CLASSIFIER

    def __call__(self, X):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        
        tokens, attention_mask, pos = preprocess_data(X, self.tokenizer, self.DEVICE, self.PAD_ID)
        with torch.no_grad():
            encoded_layers, _ = self.bert(tokens, attention_mask=attention_mask, output_all_encoded_layers=True)
            encoded_layers = torch.stack(encoded_layers, dim=1)
        vect_wordpiece = get_vect_from_pos(encoded_layers, pos)
        features = self.pooling(vect_wordpiece)
        features = torch.cat(features, dim=1)
        scores = self.mlp(features)
        return scores

    def parameters(self):
        return list(self.pooling.parameters()) + list(self.mlp.parameters())

    def save_parameters(self):
        torch.save(self.pooling.state_dict(), self.PATH_WEIGHTS_POOLING)
        torch.save(self.mlp.state_dict(), self.PATH_WEIGHTS_CLASSIFIER)

    def load_parameters_ontonotes(self):
        print('Loading OntoNotes parameters.')
        self.pooling.load_state_dict(torch.load(self.PATH_WEIGHTS_POOLING_ONTONOTES))
        self.mlp.load_state_dict(torch.load(self.PATH_WEIGHTS_CLASSIFIER_ONTONOTES))

    def load_parameters(self):
        self.pooling.load_state_dict(torch.load(self.PATH_WEIGHTS_POOLING))
        self.mlp.load_state_dict(torch.load(self.PATH_WEIGHTS_CLASSIFIER))

    def train(self):
        self.pooling.train()
        self.mlp.train()

    def eval(self):
        self.pooling.eval()
        self.mlp.eval()

import csv, os,sys, torch, logging
from tqdm import tqdm
from git import Repo
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard
from pytorch_pretrained_bert import BertTokenizer, BertModel
from neural_nets import MLP
import numpy as np

DEBUG = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', DEVICE)

FOLDER_DATA = 'gap_coreference'
if not os.path.exists(FOLDER_DATA):
    Repo.clone_from('https://github.com/antoinecollas/gap-coreference', FOLDER_DATA)
TEST_PATH = os.path.join(FOLDER_DATA, 'gap-development.tsv')

FOLDER_RESULTS = 'results'
if not os.path.exists(FOLDER_RESULTS):
    os.mkdir(FOLDER_RESULTS)
TEST_PRED_GAP_SCORER_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-scorer-development.tsv')
TEST_PRED_KAGGLE_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-kaggle-development.csv')

PATH_WEIGHTS = 'weights_classifier'

BATCH_SIZE = 1 #don't change it

if DEBUG:
    BERT_MODEL = 'bert-base-cased'
    classifier = MLP(3*768, 3) # output: nothing, A, B
else:
    BERT_MODEL = 'bert-large-cased'
    classifier = MLP(3*1024, 3)

classifier.eval()
classifier.to(DEVICE)
classifier.load_state_dict(torch.load(PATH_WEIGHTS))
print('number of parameters:', torch.nn.utils.parameters_to_vector(classifier.parameters()).shape[0])

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
pad_token = tokenizer.tokenize("[PAD]")
PAD_ID = tokenizer.convert_tokens_to_ids(pad_token)[0]

Bert = BertModel.from_pretrained(BERT_MODEL)
Bert.eval()
Bert.to(DEVICE)

loss = torch.nn.CrossEntropyLoss()
loss_values, predictions = list(), list()

# data_test = DataLoader(TEST_PATH, BATCH_SIZE, shuffle=False, debug=False)
data_test = DataLoader(TEST_PATH, BATCH_SIZE, shuffle=False, debug=DEBUG)

for X, Y in tqdm(data_test):
    tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, DEVICE, PAD_ID)

    with torch.no_grad():
        encoded_layers, _ = Bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]
        vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
        features = torch.cat(vect_wordpiece, dim=1)

        output = classifier(features)

        loss_value = loss(output, Y)
        loss_values.append(loss_value.item())
        predictions.append(np.array([X.iloc[0]['ID'], output[0][0].item(), output[0][1].item(), output[0][2].item()]))

print('Loss on development set:', np.mean(loss_values))

with open(TEST_PRED_GAP_SCORER_PATH, 'w', encoding='utf8', newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    # tsv_writer.writerow(['ID', 'A-coref', 'B-coref'])
    for prediction in predictions:
        argmax = np.argmax(prediction[1:])
        if argmax == 0:
            to_write = ['TRUE', 'FALSE']
        elif argmax == 1:
            to_write = ['FALSE', 'TRUE']
        else:
            to_write = ['FALSE', 'FALSE']
        tsv_writer.writerow([prediction[0], *to_write])

with open(TEST_PRED_KAGGLE_PATH, 'w', encoding='utf8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
    csv_writer.writerow(['ID', 'A', 'B', 'NEITHER'])
    for prediction in predictions:
        csv_writer.writerow(prediction)
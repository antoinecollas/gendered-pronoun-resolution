import csv, os,sys, torch, argparse
from tqdm import tqdm
from git import Repo
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard
from pytorch_pretrained_bert import BertTokenizer, BertModel
from neural_nets import MLP, Pooling
import numpy as np

parser = argparse.ArgumentParser(description='Testing model for coreference.')
parser.add_argument('--debug', help='Debug mode.', action='store_true')
args = parser.parse_args()
DEBUG = args.debug
if DEBUG:
    print('============ DEBUG ============')

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

PATH_WEIGHTS_POOLING = 'weights_pooling'
PATH_WEIGHTS_CLASSIFIER = 'weights_classifier'

BATCH_SIZE = 1 #don't change it
D_PROJ = 256

if DEBUG:
    BERT_MODEL = 'bert-base-uncased'
    pooling = Pooling(768, D_PROJ)
    EVALUATION_FREQUENCY = 1
else:
    BERT_MODEL = 'bert-large-uncased'
    pooling = Pooling(1024, D_PROJ)
    EVALUATION_FREQUENCY = 5

pooling.eval().to(DEVICE)
pooling.load_state_dict(torch.load(PATH_WEIGHTS_POOLING))
classifier = MLP(3*D_PROJ, 3)
classifier.eval().to(DEVICE)
classifier.load_state_dict(torch.load(PATH_WEIGHTS_CLASSIFIER))

print('number of parameters in pooling:', torch.nn.utils.parameters_to_vector(pooling.parameters()).shape[0])
print('number of parameters in classifier:', torch.nn.utils.parameters_to_vector(classifier.parameters()).shape[0])

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
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
        features = pooling(vect_wordpiece)
        features = torch.cat(features, dim=1)

        output = classifier(features)

        loss_value = loss(output, Y)
        loss_values.append(loss_value.item())
        predictions.append(np.array([X.iloc[0]['ID'], output[0][0].item(), output[0][1].item(), output[0][2].item()]))

print('Loss (cross entropy) on development set:', np.mean(loss_values))

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
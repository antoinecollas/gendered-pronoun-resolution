import os, sys, torch, argparse
from git import Repo
from pytorch_pretrained_bert import BertTokenizer, BertModel
from neural_nets import MLP, Pooling
from tensorboardX import SummaryWriter
from train import train
from test import test

class Cfg:
    pass
cfg = Cfg()  # Create an empty configuration

parser = argparse.ArgumentParser(description='Training or testing model for coreference.')
parser.add_argument('--debug', help='Debug mode.', action='store_true')
parser.add_argument('--train', help='Training mode.', action='store_true')
parser.add_argument('--test', help='Testing mode.', action='store_true')
args = parser.parse_args()
if (args.train and args.test) or (not(args.train) and not(args.test)):
    raise ValueError('You have to use --train or --test option.')
cfg.DEBUG = args.debug
cfg.TRAIN = args.train
cfg.TEST = args.test

if cfg.DEBUG:
    print('============ DEBUG ============')

cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', cfg.DEVICE)
writer = SummaryWriter()

FOLDER_DATA = 'gap_coreference'
if not os.path.exists(FOLDER_DATA):
    Repo.clone_from('https://github.com/antoinecollas/gap-coreference', FOLDER_DATA)
cfg.TRAINING_PATH = os.path.join(FOLDER_DATA, 'gap-test.tsv')
cfg.VAL_PATH = os.path.join(FOLDER_DATA, 'gap-validation.tsv')
cfg.TEST_PATH = os.path.join(FOLDER_DATA, 'gap-development.tsv')

cfg.PATH_WEIGHTS_POOLING = 'weights_pooling'
cfg.PATH_WEIGHTS_CLASSIFIER = 'weights_classifier'

FOLDER_RESULTS = 'results'
if not os.path.exists(FOLDER_RESULTS):
    os.mkdir(FOLDER_RESULTS)
cfg.TEST_PRED_GAP_SCORER_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-scorer-development.tsv')
cfg.TEST_PRED_KAGGLE_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-kaggle-development.csv')

cfg.NB_EPOCHS = 20
cfg.D_PROJ = 256

classifier = MLP(3*cfg.D_PROJ, 3)
classifier.to(cfg.DEVICE)

if cfg.DEBUG:
    cfg.BERT_MODEL = 'bert-base-uncased'
    pooling = Pooling(768, cfg.D_PROJ).to(cfg.DEVICE)
    cfg.BATCH_SIZE = 2
else:
    cfg.BERT_MODEL = 'bert-large-uncased'
    pooling = Pooling(1024, cfg.D_PROJ).to(cfg.DEVICE)
    cfg.BATCH_SIZE = 32
cfg.EVALUATION_FREQUENCY = 1

print('number of parameters in pooling:', torch.nn.utils.parameters_to_vector(pooling.parameters()).shape[0])
print('number of parameters in classifier:', torch.nn.utils.parameters_to_vector(classifier.parameters()).shape[0])

# load pre-trained bert tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL, do_lower_case=True)
pad_token = tokenizer.tokenize("[PAD]")
cfg.PAD_ID = tokenizer.convert_tokens_to_ids(pad_token)[0]

# load pretrained bert
bert = BertModel.from_pretrained(cfg.BERT_MODEL)
bert.to(cfg.DEVICE)

if cfg.TRAIN:
    train(tokenizer, bert, pooling, classifier, cfg, writer)
elif cfg.TEST:
    cfg.BATCH_SIZE = 1
    pooling.load_state_dict(torch.load(cfg.PATH_WEIGHTS_POOLING))
    classifier.load_state_dict(torch.load(cfg.PATH_WEIGHTS_CLASSIFIER))
    test(tokenizer, bert, pooling, classifier, cfg)
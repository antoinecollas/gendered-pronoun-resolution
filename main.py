import os, torch, argparse
from git import Repo
from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from test import test
from utils import DataLoader

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

FOLDER_RESULTS = 'results'
if not os.path.exists(FOLDER_RESULTS):
    os.mkdir(FOLDER_RESULTS)
cfg.TEST_PRED_GAP_SCORER_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-scorer-development.tsv')
cfg.TEST_PRED_KAGGLE_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-kaggle-development.csv')

cfg.NB_EPOCHS = 20
cfg.D_PROJ = 256
cfg.BATCH_SIZE = 2 if cfg.DEBUG else 32
cfg.EVALUATION_FREQUENCY = 1

if cfg.TRAIN:
    cfg.PATH_WEIGHTS_POOLING_LOAD = 'weights_pooling_ontonotes'
    cfg.PATH_WEIGHTS_CLASSIFIER_LOAD = 'weights_classifier_ontonotes'
    cfg.PATH_WEIGHTS_POOLING_SAVE = 'weights_pooling'
    cfg.PATH_WEIGHTS_CLASSIFIER_SAVE = 'weights_classifier'
elif cfg.TEST:
    cfg.PATH_WEIGHTS_POOLING_LOAD = 'weights_pooling'
    cfg.PATH_WEIGHTS_CLASSIFIER_LOAD = 'weights_classifier'
    cfg.PATH_WEIGHTS_POOLING_SAVE = None
    cfg.PATH_WEIGHTS_CLASSIFIER_SAVE = None

model = Model(cfg)
model.load_parameters()

print('number of parameters in pooling:', torch.nn.utils.parameters_to_vector(model.pooling.parameters()).shape[0])
print('number of parameters in mlp:', torch.nn.utils.parameters_to_vector(model.mlp.parameters()).shape[0])

if cfg.TRAIN:
    data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
    data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
    train(model, data_training, data_eval, cfg, writer)
elif cfg.TEST:
    cfg.BATCH_SIZE = 1
    data_test = DataLoader(cfg.TEST_PATH, cfg.BATCH_SIZE, shuffle=False, debug=cfg.DEBUG)
    test(model, data_test, cfg)
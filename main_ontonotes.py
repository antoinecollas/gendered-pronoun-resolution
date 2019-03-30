import os, torch, argparse
from git import Repo
from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from test import test
from utils import Ontonotes

class Cfg:
    pass
cfg = Cfg()  # Create an empty configuration

parser = argparse.ArgumentParser(description='Training or testing model for coreference.')
parser.add_argument('--debug', help='Debug mode.', action='store_true')
args = parser.parse_args()
cfg.DEBUG = args.debug

if cfg.DEBUG:
    print('============ DEBUG ============')

cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('RUNNING ON', cfg.DEVICE)
writer = SummaryWriter()

FOLDER_DATA = 'ontonotes/coref_jiant'
if not os.path.exists(FOLDER_DATA):
   raise NotImplementedError
cfg.TRAINING_PATH = os.path.join(FOLDER_DATA, 'train.json')
cfg.VAL_PATH = os.path.join(FOLDER_DATA, 'development.json')

cfg.PATH_WEIGHTS_POOLING = 'weights_pooling_ontonotes'
cfg.PATH_WEIGHTS_CLASSIFIER = 'weights_classifier_ontonotes'
cfg.PATH_WEIGHTS_POOLING_ONTONOTES = None
cfg.PATH_WEIGHTS_CLASSIFIER_ONTONOTES = None

cfg.NB_EPOCHS = 8
cfg.NB_OUTPUTS = 4
cfg.D_PROJ = 256
cfg.BATCH_SIZE = 2 if cfg.DEBUG else 32
cfg.EVALUATION_FREQUENCY = 1

cfg.PATH_WEIGHTS_POOLING_LOAD = None
cfg.PATH_WEIGHTS_CLASSIFIER_LOAD = None
cfg.PATH_WEIGHTS_POOLING_SAVE = 'weights_pooling_ontonotes'
cfg.PATH_WEIGHTS_CLASSIFIER_SAVE = 'weights_classifier_ontonotes'

model = Model(cfg)

print('number of parameters in pooling:', torch.nn.utils.parameters_to_vector(model.pooling.parameters()).shape[0])
print('number of parameters in mlp:', torch.nn.utils.parameters_to_vector(model.mlp.parameters()).shape[0])

data_training = Ontonotes(cfg.TRAINING_PATH, cfg.BATCH_SIZE, debug=cfg.DEBUG)
data_eval = Ontonotes(cfg.VAL_PATH, cfg.BATCH_SIZE, debug=cfg.DEBUG)
train(model, data_training, data_eval, cfg, writer)
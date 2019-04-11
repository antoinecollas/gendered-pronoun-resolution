import os, torch, argparse
from git import Repo
from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from test import test
from utils import DataLoader
from cfg import Cfg

parser = argparse.ArgumentParser(description='Training or testing model for coreference.')
parser.add_argument('--debug', help='Debug mode.', action='store_true')
parser.add_argument('--train', help='Training mode.', action='store_true')
parser.add_argument('--test', help='Testing mode.', action='store_true')
parser.add_argument('--use_pretrain_ontonotes', help='Use ontonotes pretrained weights.', action='store_true')
args = parser.parse_args()
if (args.train and args.test) or (not(args.train) and not(args.test)):
    raise ValueError('You have to use --train or --test option.')

cfg = Cfg(args.debug, args.train, args.test, args.use_pretrain_ontonotes)

model = Model(cfg)
writer = SummaryWriter()

print('number of parameters in BERT:', torch.nn.utils.parameters_to_vector(model.bert.parameters()).shape[0])
print('number of parameters in pooling:', torch.nn.utils.parameters_to_vector(model.pooling.parameters()).shape[0])
print('number of parameters in mlp:', torch.nn.utils.parameters_to_vector(model.mlp.parameters()).shape[0])
print('total number of parameters:', torch.nn.utils.parameters_to_vector(model.parameters()).shape[0])

if cfg.TRAIN:
    data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, endless_iterator=True, shuffle=True, debug=cfg.DEBUG)
    data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=True, debug=cfg.DEBUG)
    train(model, data_training, data_eval, cfg, writer)
elif cfg.TEST:
    data_test = DataLoader(cfg.TEST_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=False, debug=cfg.DEBUG)
    test(model, data_test, cfg)
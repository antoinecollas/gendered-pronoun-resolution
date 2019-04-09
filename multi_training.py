import torch
from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from test import test
from utils import DataLoader
from cfg import Cfg

DEBUG = True
TRAIN = True
TEST = False
ONTONOTES = False

list_cfgs = list()
cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
cfg.LR = 1e-3
list_cfgs.append((cfg, 'lr=1e-3'))

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
cfg.LR = 1e-4
list_cfgs.append((cfg, 'lr=1e-4'))

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
cfg.LR = 1e-5
list_cfgs.append((cfg, 'lr=1e-5'))

model = Model(cfg)

data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)

for cfg in list_cfgs:
    print(cfg[1])
    writer = SummaryWriter(comment=cfg[1])
    train(model, data_training, data_eval, cfg[0], writer)

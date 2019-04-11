import torch
from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from test import test
from utils import DataLoader
from cfg import Cfg

DEBUG = False
TRAIN = True
TEST = False
ONTONOTES = False

list_cfgs = list()

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
cfg.LR = 1e-5
cfg.NB_ITER = 40
cfg.EVALUATION_FREQUENCY = 4
list_cfgs.append((cfg, 'lr='+str(cfg.LR)))

for cfg in list_cfgs:
    print(cfg[1])
    writer = SummaryWriter(comment=cfg[1])
    cfg = cfg[0]
    model = Model(cfg)
    data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, endless_iterator=True, shuffle=True, debug=cfg.DEBUG)
    data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=True, debug=cfg.DEBUG)
    train(model, data_training, data_eval, cfg, writer)

from neural_nets import Model
from tensorboardX import SummaryWriter
from train import train
from utils import DataLoader
from cfg import Cfg

NB_MODELS = 5

DEBUG = False
TRAIN = True
TEST = False
ONTONOTES = False

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)

for i in range(NB_MODELS):
    print('N° model:', i)
    cfg.PATH_WEIGHTS_SAVE = 'weights_'+str(i)
    comment = '_'+str(i)
    writer = SummaryWriter(comment=comment)
    model = Model(cfg)
    data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, endless_iterator=True, shuffle=True, debug=cfg.DEBUG)
    data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=True, debug=cfg.DEBUG)
    train(model, data_training, data_eval, cfg, writer)

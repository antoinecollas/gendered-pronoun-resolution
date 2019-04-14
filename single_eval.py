from neural_nets import Model
from test import test
from utils import DataLoader
from cfg import Cfg

DEBUG = False
TRAIN = False
TEST = True
ONTONOTES = False

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
models = list()

model = Model(cfg)
data_test = DataLoader(cfg.TEST_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=False, debug=cfg.DEBUG)
test(model, data_test, cfg)
from neural_nets import Model
from test import test
from utils import DataLoader
from cfg import Cfg

NB_MODELS = 5

DEBUG = False
TRAIN = False
TEST = True
ONTONOTES = False

cfg = Cfg(DEBUG, TRAIN, TEST, ONTONOTES)
models = list()

for i in range(NB_MODELS):
    cfg.PATH_WEIGHTS_LOAD = 'weights_'+str(i)
    model = Model(cfg)
    models.append(model)
data_test = DataLoader(cfg.TEST_PATH, cfg.BATCH_SIZE, endless_iterator=False, shuffle=False, debug=cfg.DEBUG)
test(models, data_test, cfg)
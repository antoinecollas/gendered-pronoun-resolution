import torch, os
from git import Repo

class Cfg():
    def __init__(self, debug, train, test, use_pretrain_ontonotes):
        self.DEBUG = debug
        self.TRAIN = train
        self.TEST = test
        self.ONTONOTES = use_pretrain_ontonotes

        if self.DEBUG:
            print('============ DEBUG ============')

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('RUNNING ON', self.DEVICE)

        FOLDER_DATA = 'gap_coreference'
        if not os.path.exists(FOLDER_DATA):
            Repo.clone_from('https://github.com/antoinecollas/gap-coreference', FOLDER_DATA)
        self.TRAINING_PATH = os.path.join(FOLDER_DATA, 'gap-test-development.tsv')
        self.VAL_PATH = os.path.join(FOLDER_DATA, 'gap-validation.tsv')
        # self.TEST_PATH = os.path.join(FOLDER_DATA, 'gap-validation.tsv')
        self.TEST_PATH = os.path.join(FOLDER_DATA, 'test_stage_2.tsv')

        FOLDER_RESULTS = 'results'
        if not os.path.exists(FOLDER_RESULTS):
            os.mkdir(FOLDER_RESULTS)
        self.TEST_PRED_GAP_SCORER_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-scorer-development.tsv')
        self.TEST_PRED_KAGGLE_PATH = os.path.join(FOLDER_RESULTS, 'gap-pred-kaggle-development.csv')

        self.ADD_FEATURES = False
        self.TRAIN_END_2_END = True
        self.NB_OUTPUTS = 3
        if self.DEBUG:
            self.NB_ITER = 10
            self.EVALUATION_FREQUENCY = 5
        else:
            self.NB_ITER = 2000
            self.EVALUATION_FREQUENCY = 100
        self.DROPOUT = 0.2
        self.D_PROJ = 256
        self.D_HID = 512
        self.LR = 1e-5

        if self.TRAIN:
            self.PATH_WEIGHTS_LOAD = 'weights_ontonotes'
            self.PATH_WEIGHTS_SAVE = 'weights'
            self.BATCH_SIZE = 2 if (self.DEBUG or self.TRAIN_END_2_END) else 32
        elif self.TEST:
            self.PATH_WEIGHTS_LOAD = 'weights'
            self.PATH_WEIGHTS_SAVE = None
            self.BATCH_SIZE = 1
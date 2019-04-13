import torch, csv
from tqdm import tqdm
from utils import compute_word_pos, pad, get_vect_from_pos, print_tensorboard
import numpy as np
import torch.nn as nn

def test(models, data_test, cfg):
    if not isinstance(models, list):
        models = [models]
    
    predictions = np.zeros((len(models), len(data_test), 3))
    for i, model in enumerate(models):
        model.eval()
        softmax = nn.Softmax(dim=1)
        ids = list()

        for j, (X, Y) in enumerate(tqdm(data_test)):
            Y = Y.to(cfg.DEVICE)
            output = model(X)
            output = softmax(output)
            ids.append(X.iloc[0]['ID'])
            predictions[i, j, :] = [output[0][0].item(), output[0][1].item(), output[0][2].item()]
    predictions = predictions.mean(axis=0)

    with open(cfg.TEST_PRED_GAP_SCORER_PATH, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        # tsv_writer.writerow(['ID', 'A-coref', 'B-coref'])
        for id_, prediction in zip(ids, predictions):
            argmax = np.argmax(prediction)
            if argmax == 0:
                to_write = ['TRUE', 'FALSE']
            elif argmax == 1:
                to_write = ['FALSE', 'TRUE']
            else:
                to_write = ['FALSE', 'FALSE']
            tsv_writer.writerow([id_, *to_write])

    with open(cfg.TEST_PRED_KAGGLE_PATH, 'w', encoding='utf8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        csv_writer.writerow(['ID', 'A', 'B', 'NEITHER'])
        for id_, prediction in zip(ids, predictions):
            csv_writer.writerow([id_, *prediction])
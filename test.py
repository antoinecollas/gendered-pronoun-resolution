import sys, torch, csv
from tqdm import tqdm
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard
import numpy as np
import torch.nn as nn

def test(model, cfg):
    model.eval()
    softmax = nn.Softmax(dim=1)
    predictions = list()
    data_test = DataLoader(cfg.TEST_PATH, cfg.BATCH_SIZE, shuffle=False, debug=cfg.DEBUG)

    for X, Y in tqdm(data_test):
        Y.to(cfg.DEVICE)
        output = model(X)
        output = softmax(output)
        predictions.append(np.array([X.iloc[0]['ID'], output[0][0].item(), output[0][1].item(), output[0][2].item()]))

    with open(cfg.TEST_PRED_GAP_SCORER_PATH, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        # tsv_writer.writerow(['ID', 'A-coref', 'B-coref'])
        for prediction in predictions:
            argmax = np.argmax(prediction[1:])
            if argmax == 0:
                to_write = ['TRUE', 'FALSE']
            elif argmax == 1:
                to_write = ['FALSE', 'TRUE']
            else:
                to_write = ['FALSE', 'FALSE']
            tsv_writer.writerow([prediction[0], *to_write])

    with open(cfg.TEST_PRED_KAGGLE_PATH, 'w', encoding='utf8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        csv_writer.writerow(['ID', 'A', 'B', 'NEITHER'])
        for prediction in predictions:
            csv_writer.writerow(prediction)
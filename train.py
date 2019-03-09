import sys, torch
from tqdm import tqdm
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard, log_loss
import numpy as np

def train(tokenizer, bert, pooling, classifier, cfg, tensorboard_writer):
    bert.eval()

    loss = torch.nn.CrossEntropyLoss()
    loss_values, loss_values_eval, log_loss_values_eval = list(), list(), list()
    optimizer = torch.optim.Adam(list(pooling.parameters()) + list(classifier.parameters()), lr = 0.0001)

    for epoch in tqdm(range(cfg.NB_EPOCHS)):
        data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
        pooling.train()
        classifier.train()

        for X, Y in data_training:
            tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, cfg.DEVICE, cfg.PAD_ID)

            with torch.no_grad():
                encoded_layers, _ = bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]
            vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
            features = pooling(vect_wordpiece)
            features = torch.cat(features, dim=1)

            output = classifier(features)
            
            optimizer.zero_grad()
            output = loss(output, Y)
            loss_values.append(output.item())
            output.backward()
            torch.nn.utils.clip_grad_norm_(list(pooling.parameters()) + list(classifier.parameters()), max_norm=5, norm_type=2)
            optimizer.step()

        if epoch%cfg.EVALUATION_FREQUENCY == 0:
            data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
            pooling.eval()
            classifier.eval()

            for X, Y in data_eval:
                tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, cfg.DEVICE, cfg.PAD_ID)
                
                with torch.no_grad():
                    encoded_layers, _ = bert(tokens, attention_mask=attention_mask) #list of [bs, max_len, 768]
                vect_wordpiece = get_vect_from_pos(encoded_layers[len(encoded_layers)-1], pos)
                features = pooling(vect_wordpiece)
                features = torch.cat(features, dim=1)

                with torch.no_grad():
                    output = classifier(features)

                loss_value = loss(output, Y)
                loss_values_eval.append(loss_value.item())
                log_loss_values_eval.append(log_loss(output, Y).item())

            # the losses are not totally correct because it assumes that all batch have the same size whereas the last one is often smaller
            scalars = {
                'training/cross_entropy': np.mean(loss_values),
                'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(list(pooling.parameters()) + list(classifier.parameters())), p=2),
                'eval/cross_entropy'  : np.mean(loss_values_eval),
                'eval/log_loss'  : np.mean(log_loss_values_eval)
            }
            print_tensorboard(tensorboard_writer, scalars, epoch)
            loss_values, loss_values_eval, log_loss_values_eval = list(), list(), list()
            torch.save(pooling.state_dict(), cfg.PATH_WEIGHTS_POOLING)
            torch.save(classifier.state_dict(), cfg.PATH_WEIGHTS_CLASSIFIER)
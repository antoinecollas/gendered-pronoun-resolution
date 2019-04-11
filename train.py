import torch, sys
from tqdm import tqdm
from utils import compute_word_pos, pad, get_vect_from_pos, print_tensorboard
from sklearn.metrics import f1_score
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np

def train(model, data_training, data_eval, cfg, tensorboard_writer):
    loss = torch.nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=cfg.LR,
                                warmup=0.1,
                                t_total=cfg.NB_ITER)

    output_values, Y_true, grad_norm = list(), list(), list()

    for i, (X, Y) in zip(tqdm(range(cfg.NB_ITER)), data_training):
        model.train()

        Y = Y.to(cfg.DEVICE)
        output_model = model(X)
        optimizer.zero_grad()
        output = loss(output_model, Y)
        output.backward()
        optimizer.step()
        
        output_values.append(output_model.detach_())
        Y_true.append(Y)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm.append(total_norm ** (1./2))

        if (i+1)%cfg.EVALUATION_FREQUENCY == 0:
            model.eval()

            output_values = torch.cat(output_values)
            Y_true = torch.cat(Y_true)
            loss_value = loss(output_values, Y_true)

            output_values, Y_true = list(), list()

            for X, Y in data_eval:
                Y = Y.to(cfg.DEVICE)
                output = model(X)
                output_values.append(output.detach_())
                Y_true.append(Y)

            output_values = torch.cat(output_values)
            Y_true = torch.cat(Y_true)
            loss_value_eval = loss(output_values, Y_true)

            Y_pred = torch.argmax(output_values, dim=1)
            gold_A = (Y_true == 1)
            system_A = (Y_pred == 1)
            gold_B = (Y_true == 2)
            system_B = (Y_pred == 2)
            gold = torch.cat((gold_A, gold_B))
            system = torch.cat((system_A, system_B))
            f1 = f1_score(gold.cpu(), system.cpu())

            scalars = {
                'training/cross_entropy': loss_value,
                'training/gradient_norm': np.mean(grad_norm),
                'eval/cross_entropy': loss_value_eval,
                'eval/f1': f1,
            }
            print_tensorboard(tensorboard_writer, scalars, i)
            model.save_parameters()

            output_values, Y_true, grad_norm = list(), list(), list()
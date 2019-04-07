import torch
from tqdm import tqdm
from utils import compute_word_pos, pad, get_vect_from_pos, print_tensorboard
from sklearn.metrics import f1_score

def train(model, data_training, data_eval, cfg, tensorboard_writer):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in tqdm(range(cfg.NB_EPOCHS)):
        model.train()
        output_values, Y_true = list(), list()

        for X, Y in data_training:
            Y = Y.to(cfg.DEVICE)
            output_model = model(X)
            optimizer.zero_grad()
            output = loss(output_model, Y)
            output.backward()
            # for p in model.parameters():
            #     if p.grad is not None:
            #         print(p.grad.shape)
            #     else:
            #         print('None grad:', p.shape)
            optimizer.step()

            output_values.append(output_model.detach_())
            Y_true.append(Y)

        output_values = torch.cat(output_values)
        Y_true = torch.cat(Y_true)
        loss_value = loss(output_values, Y_true)

        if epoch%cfg.EVALUATION_FREQUENCY == 0:
            model.eval()
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
                'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(model.parameters()), p=2),
                'eval/cross_entropy'  : loss_value_eval,
                'eval/f1'  : f1,
            }
            print_tensorboard(tensorboard_writer, scalars, epoch)
            model.save_parameters()
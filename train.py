import torch
from tqdm import tqdm
from utils import compute_word_pos, pad, get_vect_from_pos, print_tensorboard

def train(model, data_training, data_eval, cfg, tensorboard_writer):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in tqdm(range(cfg.NB_EPOCHS)):
        model.train()
        output_values, Y_values = list(), list()

        for X, Y in data_training:
            Y = Y.to(cfg.DEVICE)
            output_model = model(X)
            optimizer.zero_grad()
            output = loss(output_model, Y)
            output.backward()
            optimizer.step()

            output_values.append(output_model.detach_())
            Y_values.append(Y)

        output_values = torch.cat(output_values)
        Y_values = torch.cat(Y_values)
        loss_value = loss(output_values, Y_values)

        if epoch%cfg.EVALUATION_FREQUENCY == 0:
            model.eval()
            output_values, Y_values = list(), list()

            for X, Y in data_eval:
                Y = Y.to(cfg.DEVICE)
                output = model(X)
                output_values.append(output.detach_())
                Y_values.append(Y)

            output_values = torch.cat(output_values)
            Y_values = torch.cat(Y_values)
            loss_value_eval = loss(output_values, Y_values)

            scalars = {
                'training/cross_entropy': loss_value,
                'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(model.parameters()), p=2),
                'eval/cross_entropy'  : loss_value_eval,
            }
            print_tensorboard(tensorboard_writer, scalars, epoch)
            model.save_parameters()
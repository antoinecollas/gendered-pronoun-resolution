import sys, torch
from tqdm import tqdm
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard

def train(model, cfg, tensorboard_writer):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in tqdm(range(cfg.NB_EPOCHS)):
        model.train()
        data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
        output_values, Y_values = list(), list()

        for X, Y in data_training:
            Y = Y.to(cfg.DEVICE)
            output = model(X)
            output_values.append(output)
            Y_values.append(Y)
            optimizer.zero_grad()
            output = loss(output, Y)
            output.backward()
            # torch.nn.utils.clip_grad_norm_(list(pooling.parameters()) + list(classifier.parameters()), max_norm=5, norm_type=2)
            optimizer.step()

        output_values = torch.cat(output_values)
        Y_values = torch.cat(Y_values)
        loss_value = loss(output_values, Y_values)

        if epoch%cfg.EVALUATION_FREQUENCY == 0:
            data_eval = DataLoader(cfg.VAL_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
            model.eval()
            output_values, Y_values = list(), list()

            for X, Y in data_eval:
                Y = Y.to(cfg.DEVICE)
                output = model(X)
                output_values.append(output)
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
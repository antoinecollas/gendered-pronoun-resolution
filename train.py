import sys, torch
from tqdm import tqdm
from utils import DataLoader, compute_word_pos, pad, get_vect_from_pos, preprocess_data, print_tensorboard

def train(tokenizer, bert, pooling, classifier, cfg, tensorboard_writer):
    bert.eval()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(pooling.parameters()) + list(classifier.parameters()), lr = 0.0001)

    for epoch in tqdm(range(cfg.NB_EPOCHS)):
        data_training = DataLoader(cfg.TRAINING_PATH, cfg.BATCH_SIZE, shuffle=True, debug=cfg.DEBUG)
        pooling.train()
        classifier.train()
        output_values, Y_values = list(), list()

        for X, Y in data_training:
            tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, cfg.DEVICE, cfg.PAD_ID)

            with torch.no_grad():
                encoded_layers, _ = bert(tokens, attention_mask=attention_mask, output_all_encoded_layers=True)
                encoded_layers = torch.stack(encoded_layers, dim=1)
            vect_wordpiece = get_vect_from_pos(encoded_layers, pos)
            features = pooling(vect_wordpiece)
            features = torch.cat(features, dim=1)

            output = classifier(features)
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
            pooling.eval()
            classifier.eval()
            output_values, Y_values = list(), list()
            for X, Y in data_eval:
                tokens, Y, attention_mask, pos = preprocess_data(X, Y, tokenizer, cfg.DEVICE, cfg.PAD_ID)

                with torch.no_grad():
                    encoded_layers, _ = bert(tokens, attention_mask=attention_mask, output_all_encoded_layers=True)
                    encoded_layers = torch.stack(encoded_layers, dim=1)
                    vect_wordpiece = get_vect_from_pos(encoded_layers, pos)
                    features = pooling(vect_wordpiece)
                    features = torch.cat(features, dim=1)
                    output = classifier(features)
                
                output_values.append(output)
                Y_values.append(Y)

            output_values = torch.cat(output_values)
            Y_values = torch.cat(Y_values)
            loss_value_eval = loss(output_values, Y_values)

            scalars = {
                'training/cross_entropy': loss_value,
                'training/gradient_norm': torch.norm(torch.nn.utils.parameters_to_vector(list(pooling.parameters()) + list(classifier.parameters())), p=2),
                'eval/cross_entropy'  : loss_value_eval,
            }
            print_tensorboard(tensorboard_writer, scalars, epoch)
            torch.save(pooling.state_dict(), cfg.PATH_WEIGHTS_POOLING)
            torch.save(classifier.state_dict(), cfg.PATH_WEIGHTS_CLASSIFIER)
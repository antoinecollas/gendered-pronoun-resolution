import jsonlines, os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class DataLoader():
    def __init__(self, path, batch_size, shuffle=False, debug=False):
        self.data = pd.read_csv(path, delimiter='\t')
        if debug:
            self.data = self.data.iloc[0:10]
        self.batch_size = batch_size
        self.current_idx = 0
        if shuffle:
            order = np.arange(len(self.data))
            np.random.shuffle(order)
            self.data = self.data.iloc[order]

    def __len__(self):
        return len(self.data)

    def get_col_names(self):
        return list(self.data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx>=len(self.data):
            self.current_idx = 0
            raise StopIteration
        temp = self.current_idx
        self.current_idx += self.batch_size
        X =  self.data[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset']].iloc[temp:self.current_idx]
        Y =  self.data[['A-coref', 'B-coref']].iloc[temp:self.current_idx]
        temp = torch.Tensor(Y.values.astype(int))
        Y = torch.zeros((Y.shape[0], Y.shape[1]+1)) 
        Y[:,0:2] = temp
        Y[:,2] = 1 - Y.sum(dim=1)
        Y = torch.argmax(Y, dim=1)
        return X, Y

def compute_word_pos(raw_text, wordpiece, word, offset):
    '''
    It computes the word position in the tokenized text from the raw text, the actual word and the offset.

    /!\ It assumes that every letters from the raw text are in word piece text.
    '''
    word_pos_start, wp_pos = 1, 0
    wp_ch = wordpiece[word_pos_start][wp_pos]
    for i in range(offset):
        raw_ch = raw_text[i].lower()
        if raw_ch==' ':
            continue
        while wp_ch != raw_ch:
            if wp_pos < len(wordpiece[word_pos_start])-1:
                wp_pos += 1
            else:
                word_pos_start += 1
                wp_pos = 0
            wp_ch = wordpiece[word_pos_start][wp_pos]
        
        if wp_pos < len(wordpiece[word_pos_start])-1:
            wp_pos += 1
        else:
            word_pos_start += 1
            wp_pos = 0
        wp_ch = wordpiece[word_pos_start][wp_pos]

    word_pos_end = word_pos_start
    wp_pos = 0
    wp_ch = wordpiece[word_pos_end][wp_pos]
    for raw_ch in word:
        raw_ch = raw_ch.lower()
        if raw_ch == ' ':
            continue
        while wp_ch != raw_ch:
            if wp_pos < len(wordpiece[word_pos_end])-1:
                wp_pos += 1
            else:
                word_pos_end += 1
                wp_pos = 0
            wp_ch = wordpiece[word_pos_end][wp_pos]
        
        if wp_pos < len(wordpiece[word_pos_end])-1:
            wp_pos += 1
        else:
            word_pos_end += 1
            wp_pos = 0
        wp_ch = wordpiece[word_pos_end][wp_pos]

    return [word_pos_start, word_pos_end]

def pad(tokens, pad_id):
    max_len = 0
    for i in range(len(tokens)):
        if len(tokens[i]) > max_len:
            max_len = len(tokens[i])
    
    for i in range(len(tokens)):
        while len(tokens[i]) < max_len:
            tokens[i].append(pad_id)
    
    return tokens

def get_vect_from_pos(encoded_layers, pos):
    '''
    encoded_layers: Tensor of shape [bs, nb_layers, max_len, hidden_size]
    pos: Tensor: positions of pronoun, A, B. Shape: [bs, 3, 2]
    '''
    def get_vect(encoded_layers, pos):
        vect = list()
        for i in range(pos.shape[0]):
            vect.append(encoded_layers.new_zeros(encoded_layers.shape[1], pos[i,1]-pos[i,0]+1, encoded_layers.shape[3]))
            vect[len(vect)-1] = encoded_layers[i, :, pos[i,0]:pos[i,1], :]
            vect[len(vect)-1] = vect[len(vect)-1].reshape([vect[len(vect)-1].shape[0]*vect[len(vect)-1].shape[1], vect[len(vect)-1].shape[2]])
        return vect
    
    pos_pronouns = pos[:,0,:]
    vect_pronoun = get_vect(encoded_layers, pos_pronouns)

    pos_A = pos[:,1,:]
    vect_A = get_vect(encoded_layers, pos_A)

    pos_B = pos[:,2,:]
    vect_B = get_vect(encoded_layers, pos_B)

    return [vect_pronoun, vect_A, vect_B]

def preprocess_data(X, tokenizer, device, pad_id):
    tokens, pos = list(), list()
    for row in X.itertuples(index=False):
        tokenized_text = ['[CLS]'] + tokenizer.tokenize(row.Text) + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        pos_pronoun = compute_word_pos(row.Text, tokenized_text, row.Pronoun, row._3)
        pos_A = compute_word_pos(row.Text, tokenized_text, row.A, row._5)
        pos_B = compute_word_pos(row.Text, tokenized_text, row.B, row._7)
        pos.append([pos_pronoun, pos_A, pos_B])
    pos = torch.Tensor(pos).long()

    tokens = pad(tokens, pad_id)
    tokens = torch.tensor(tokens).to(device)
    attention_mask = torch.ones(tokens.shape).to(device)
    attention_mask[tokens==pad_id] = 0

    return [tokens, attention_mask, pos]

def print_tensorboard(writer, scalars, epoch):
    for key, value in scalars.items():
        writer.add_scalar(key, value, epoch)


class ReaderOntoNotes():
    def __init__(self, path, shuffle=True):
        self.texts, self.targets = list(), list()
        with jsonlines.open(path) as reader:
            for obj in reader:
                self.texts.append(obj['text'])
                self.targets.append(obj['targets'])
        self.texts = np.array(self.texts)
        self.targets = np.array(self.targets)
        order = np.arange(self.texts.shape[0])
        if shuffle:
            np.random.shuffle(order)
        self.texts = self.texts[order]
        self.targets = self.targets[order]
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx>=len(self.texts):
            self.current_idx = 0
            raise StopIteration
        texts = self.texts[self.current_idx]
        targets = self.targets[self.current_idx]
        self.current_idx += 1
        return texts, targets

    def __len__(self):
        return len(self.texts)

class Ontonotes():
    def __init__(self, path, batch_size, debug=False):
        PRONOUNS = ['she', 'her', 'hers', 'he', 'his', 'him']
        ontonotes = ReaderOntoNotes(path)
        all_texts, all_targets = list(), list()
        for text, targets in ontonotes:
            if len(targets)>0:
                text = np.array(text.split(' '))
                new_targets = list()
                for target in targets:
                    span1 = ' '.join(text[target['span1'][0]:target['span1'][1]])
                    span2 = ' '.join(text[target['span2'][0]:target['span2'][1]])
                    pronoun = False
                    if (span1.lower() in PRONOUNS) and (span2.lower() not in PRONOUNS):
                        pronoun = span1
                        pronoun_idx = target['span1']
                        noun_idx = target['span2']
                    if (span2.lower() in PRONOUNS) and (span1.lower() not in PRONOUNS):
                        pronoun = span2
                        pronoun_idx = target['span2']
                        noun_idx = target['span1']
                        
                    if pronoun:
                        appended = False
                        for new_target in new_targets:
                            condition = ((new_target['pronoun'][0] == pronoun_idx[0]) and (new_target['pronoun'][1] == pronoun_idx[1]))
                            if condition:
                                new_target['noun'].append(noun_idx)
                                new_target['label'].append(target['label'])
                                appended = True
                                
                        if not appended:
                            obj = {
                                'pronoun': pronoun_idx,
                                'noun': [noun_idx],
                                'label': [target['label']]
                            }
                            new_targets.append(obj)

                if len(new_targets)>0:
                    temp = list()
                    for new_target in new_targets:
                        if len(new_target['label'])>1:
                            temp.append(new_target)
                    if len(temp)>0:
                        all_texts.append(text)
                        all_targets.append(temp)
        
        self.texts = np.array(all_texts)
        self.targets = np.array(all_targets)
        if debug:
            self.texts = self.texts[0:10]
            self.targets = self.targets[0:10]
        self.batch_size = batch_size
        self.current_idx = 0

        order = np.arange(len(self.texts))
        np.random.shuffle(order)
        self.texts = self.texts[order]
        self.targets = self.targets[order]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx>=len(self.texts):
            self.current_idx = 0
            raise StopIteration
        temp = self.current_idx
        self.current_idx += self.batch_size
        texts = self.texts[temp:self.current_idx]
        targets = self.targets[temp:self.current_idx]

        columns = ['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset']
        X = pd.DataFrame(data=None, index=None, columns=columns, dtype=None, copy=False)
        Y = torch.zeros(len(texts)).long()

        for i, (text, target) in enumerate(zip(texts, targets)):
            # X
            text = np.array(text)

            # get full text
            string_text = ' '.join(text)
            
            # select a random target
            id_target = np.random.randint(low=0, high=len(target))
            target = target[id_target]

            # get pronoun
            string_pronoun = text[target['pronoun'][0]:target['pronoun'][1]]
            string_pronoun = ' '.join(string_pronoun)

            # get pronoun offset
            offset_pronoun = 0
            for word in np.array(text)[0:target['pronoun'][0]]:
                offset_pronoun += len(word)+1 # +1 for space char 
            
            # sort nouns by order of appearance in text and select 2 random nouns
            target['noun'].sort(key=lambda x: (x[0], x[1]))
            id_nouns = np.sort(np.random.randint(low=0, high=len(target['noun']), size=(2)))
            labels = [int(target['label'][id_nouns[0]]), int(target['label'][id_nouns[1]])]
            noun_A = target['noun'][id_nouns[0]]
            string_A = text[noun_A[0]:noun_A[1]]
            string_A = ' '.join(string_A)
            noun_B = target['noun'][id_nouns[1]]
            string_B = text[noun_B[0]:noun_B[1]]
            string_B = ' '.join(string_B)

            # get nouns offsets
            offset_A = 0
            for word in np.array(text)[0:noun_A[0]]:
                offset_A += len(word)+1 # +1 for space char 
            offset_B = 0
            for word in np.array(text)[0:noun_B[0]]:
                offset_B += len(word)+1 # +1 for space char 

            X.loc[i] = ['NA', string_text, string_pronoun, offset_pronoun, string_A, offset_A, string_B, offset_B]

            # Y
            if (labels[0] == 1) and (labels[1] == 0):
                Y[i] = 1
            elif (labels[0] == 0) and (labels[1] == 1):
                Y[i] = 2
            elif (labels[0] == 1) and (labels[1] == 1):
                Y[i] = 3

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)

        return X, Y

    def __len__(self):
        return len(self.texts)
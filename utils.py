import pandas, sys
import numpy as np
import torch

class DataLoader():
	def __init__(self, path, batch_size, shuffle=False, debug=False):
		self.data = pandas.read_csv(path, delimiter='\t')
		if debug:
			self.data = self.data.iloc[0:10]
		self.batch_size = batch_size
		self.current_idx = 0
		self.order = np.arange(len(self.data))
		if shuffle:
			np.random.shuffle(self.order)

	def __len__(self):
		return len(self.data)

	def get_col_names(self):
		return list(self.data)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		if self.current_idx>=len(self.data):
			raise StopIteration
		temp = self.current_idx
		self.current_idx += self.batch_size
		indices = self.order[temp:self.current_idx]
		X =  self.data[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset']].iloc[indices]
		Y =  self.data[['A-coref', 'B-coref']].iloc[indices]
		return X, Y

def compute_word_pos(raw_text, wordpiece, word, offset):
	'''
	It computes the word position in the tokenized text from the raw text, the actual word and the offset.

	/!\ It assumes that every letters from the raw text are in word piece text.
	'''
	word_pos_start, wp_pos = 0, 0
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

def get_vect_from_pos(encoded_layer, pos):
	'''
	encoded_layer: Tensor: encoded layer of shape [bs, max_len, hidden_size]
	pos: Tensor: positions of pronoun, A, B. Shape: [bs, 3, 2]
	'''
	def get_vect(encoded_layer, pos):
		vect = list()
		for i in range(pos.shape[0]):
			vect.append(encoded_layer.new_zeros(pos[i,1]-pos[i,0]+1, encoded_layer.shape[2]))
			vect[len(vect)-1] = encoded_layer[i, pos[i,0]:pos[i,1]]
		return vect
	
	pos_pronouns = pos[:,0,:]
	vect_pronoun = get_vect(encoded_layer, pos_pronouns)
	pos_A = pos[:,1,:]
	vect_A = get_vect(encoded_layer, pos_A)
	pos_B = pos[:,2,:]
	vect_B = get_vect(encoded_layer, pos_B)

	return [vect_pronoun, vect_A, vect_B]

def preprocess_data(X, Y, tokenizer, device, pad_id):
	temp = torch.Tensor(Y.values.astype(int))
	Y = torch.zeros((Y.shape[0], Y.shape[1]+1)) 
	Y[:,0:2] = temp
	Y[:,2] = 1 - Y.sum(dim=1)
	Y = torch.argmax(Y, dim=1).to(device)

	tokens, pos = list(), list()
	for row in X.itertuples(index=False):
		tokenized_text = tokenizer.tokenize(row.Text)
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

	return [tokens, Y, attention_mask, pos]

def print_tensorboard(writer, scalars, epoch):
	for key, value in scalars.items():
		writer.add_scalar(key, value, epoch)

def log_loss(p_pred, Y_true):
	log_p = torch.log(torch.max(torch.min(p_pred, p_pred.new_ones(p_pred.shape)*(1-10**(-15))), p_pred.new_ones(p_pred.shape)*10**(-15)))
	Y_true = Y_true.unsqueeze(1)
	y_onehot = log_p.new_zeros(log_p.shape)
	y_onehot.scatter_(1, Y_true, 1)
	log_loss_value = log_p*y_onehot
	log_loss_value = - torch.sum(log_loss_value) / p_pred.shape[0]
	return log_loss_value.item()
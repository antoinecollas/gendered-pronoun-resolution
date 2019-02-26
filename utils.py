import pandas, sys
import numpy as np
from pytorch_pretrained_bert import BertTokenizer

class DataLoader():
	def __init__(self, path, batch_size, shuffle=False):
		self.data = pandas.read_csv(path, delimiter='\t')
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
		if self.current_idx>len(self.data):
			raise StopIteration
		temp = self.current_idx
		self.current_idx += self.batch_size
		indices = self.order[temp:self.current_idx]
		X =  self.data[['Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset']].iloc[indices]
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
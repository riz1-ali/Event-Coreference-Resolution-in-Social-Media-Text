from torch.utils import data
from torch.nn.utils import rnn
import torch
from gensim.models import Word2Vec

class dataset(data.Dataset):
	def __init__(self,input):
		self.input = input
	def __len__(self):
		return len(self.input)
	def __getitem__(self,index):
		return torch.Tensor(self.input)

def collate_fn(batch):
	return rnn.pack_sequence(batch)

if __name__ == '__main__':
	model = Word2Vec.load("word2vec.model")
	a = model.wv.vocab
	b = [i for i in a.keys()]
	b = b[-70:]
	data_ = [model.wv[i] for i in b]
	data_ = dataset(data_)
	loader = data.DataLoader(data_, batch_size=32, collate_fn=collate_fn, shuffle=True)
	# for i in loader:
	# 	print(i)
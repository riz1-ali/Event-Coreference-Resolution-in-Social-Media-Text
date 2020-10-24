from torch.utils import data
from torch.nn.utils import rnn
import torch
from gensim.models import Word2Vec

class dataset(data.Dataset):
	def __init__(self,tweet_pairs,distance_vectors):
		self.tweet_pairs = tweet_pairs
		self.distance_vectors = distance_vectors
		model = Word2Vec.load("word2vec.model")
		self.word2vec = model.wv.vocab
	def __len__(self):
		return len(self.tweet_pairs)
	def convert_tweet(self,tweet):
		return [ self.word2vec[word.strip()] for word in tweet.split()]
	def __getitem__(self,index):
		return torch.Tensor([self.convert_tweet(self.tweet_pairs[index][0]),
			self.convert_tweet(self.tweet_pairs[index][1])]),
		torch.Tensor(self.distance_vectors)


def collate_fn(batch):
	return rnn.pack_sequence(batch)

data_ = []
file_path = './generated_dataset.txt'
with open(file_path,'r') as f:
	for i in f:
		j = i.strip('\n').split('\t')
		data_.append(j[-1])
f = open('./tweet_pairs.txt','r')
tweet_pairs = eval(f.read())
f = open('./distance_vectors.txt','r')
distance_vectors = eval(f.read())
tweet_pair_data = [[data_[i[0]],data_[i[1]]] for i in tweet_pairs]
distance_vector_data = [[distance_vectors[i[0]],distance_vectors[i[1]]] for i in tweet_pairs]
# loader = data.DataLoader(data_, batch_size=32, collate_fn=collate_fn, shuffle=True)
dataset_ = dataset(tweet_pair_data,distance_vector_data)

from torch.utils import data
from torch.nn.utils import rnn
import torch
from gensim.models import Word2Vec
import pickle
import tokenizer as tk

class dataset(data.Dataset):
	def __init__(self, tweet_pairs, distance_vectors):
		self.tweet_pairs = tweet_pairs
		self.distance_vectors = distance_vectors
		self.vocab = None
		with open('./vocab.pkl', 'rb') as f:
			self.vocab = pickle.load(f)

	def __len__(self):
		return len(self.tweet_pairs)

	def convert_tweet(self, tweet):
		return [self.vocab.stoi[word.strip()] for word in tk.tokenize(tweet)]

	def __getitem__(self, index):
		return torch.Tensor(self.convert_tweet(self.tweet_pairs[index][0])), torch.Tensor(self.convert_tweet(
			self.tweet_pairs[index][1])), torch.Tensor(self.distance_vectors[index][0]), torch.Tensor(self.distance_vectors[index][1])


def collate_fn(batch):
	return rnn.pack_sequence(batch)


data_ = []
file_path = './generated_dataset.txt'
with open(file_path, 'r') as f:
	for i in f:
		j = i.strip('\n').split('\t')
		data_.append(j[-1].lower())

tweet_pairs, distance_vectors = [], []
with open('./tweet_pairs.pkl', 'rb') as f:
	tweet_pairs = pickle.load(f)
with open('./distance_vectors.pkl', 'rb') as f:
	distance_vectors = pickle.load(f)

tweet_pair_data = [[data_[i[0]], data_[i[1]]] for i in tweet_pairs]
distance_vector_data = [[distance_vectors[i[0]],
						 distance_vectors[i[1]]] for i in tweet_pairs]

dataset_ = dataset(tweet_pair_data, distance_vector_data)
loader = data.DataLoader(dataset_, batch_size=32, collate_fn=collate_fn, shuffle=True)
for i in loader:
	print(i)
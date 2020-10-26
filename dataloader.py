from torch.utils import data
from torch.nn.utils import rnn
import torch
from gensim.models import Word2Vec
import pickle
import tokenizer as tk
from tqdm import tqdm

class dataset(data.Dataset):
	def __init__(self, tweet_pairs, distance_vectors, trigger_word_pos):
		self.tweet_pairs = tweet_pairs
		self.distance_vectors = distance_vectors
		self.vocab = None
		with open('./vocab.pkl', 'rb') as f:
			self.vocab = pickle.load(f)
		self.trigger_word_pos = trigger_word_pos

	def __len__(self):
		return len(self.tweet_pairs)

	def convert_tweet(self, tweet):
		return [self.vocab.stoi[word.strip()] for word in tk.tokenize(tweet)]

	def __getitem__(self, index):
		return torch.Tensor(self.convert_tweet(self.tweet_pairs[index][0])), torch.Tensor(self.convert_tweet(
			self.tweet_pairs[index][1])), torch.Tensor(self.distance_vectors[index][0]), torch.Tensor(
				self.distance_vectors[index][1]), torch.Tensor(self.trigger_word_pos[index][0]), torch.Tensor(
					self.trigger_word_pos[index][1]) 


def collate_fn(batch):
	tweet1s = []
	tweet2s = []
	distance1s = []
	distance2s = []
	pos1s = []
	pos2s = []
	for data in batch:
		tweet1s.append(data[0])
		tweet2s.append(data[1])
		distance1s.append(data[2])
		distance2s.append(data[3])
		pos1s.append(data[4].tolist())
		pos2s.append(data[5].tolist())
	
	packed_tweet1s = rnn.pack_sequence(tweet1s, enforce_sorted=False)
	packed_tweet2s = rnn.pack_sequence(tweet2s, enforce_sorted=False)
	packed_distance1s = rnn.pack_sequence(distance1s, enforce_sorted=False)
	packed_distance2s = rnn.pack_sequence(distance2s, enforce_sorted=False)

	padded_tweet1s, _ = rnn.pad_packed_sequence(packed_tweet1s, batch_first=True)
	padded_tweet2s, _ = rnn.pad_packed_sequence(packed_tweet2s, batch_first=True)
	padded_distance1s, _ = rnn.pad_packed_sequence(packed_distance1s, batch_first=True)
	padded_distance2s, _ = rnn.pad_packed_sequence(packed_distance2s, batch_first=True)
	return (
		padded_tweet1s, 
		padded_tweet2s, 
		padded_distance1s, 
		padded_distance2s,
		torch.Tensor(pos1s),
		torch.Tensor(pos2s)
		)


if __name__ == "__main__":
	data_ = []
	file_path = './FinalDataset.csv'
	with open(file_path, 'r') as f:
		for i in f:
			j = i.strip('\n').split('\t')
			data_.append(j[-1].lower())

	tweet_pairs, distance_vectors = [], []
	with open('./tweet_pairs.pkl', 'rb') as f:
		tweet_pairs = pickle.load(f)
	with open('./distance_vectors.pkl', 'rb') as f:
		distance_vectors = pickle.load(f)
	with open("./trigger_word_pos.pkl", 'rb') as f:
		trigger_word_pos = pickle.load(f)

	tweet_pair_data = [[data_[i[0]], data_[i[1]]] for i in tweet_pairs]
	distance_vector_data = [[distance_vectors[i[0]],
							 distance_vectors[i[1]]] for i in tweet_pairs]
	trigger_word_pos_data = [[trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in tweet_pairs]
	dataset_ = dataset(tweet_pair_data, distance_vector_data, trigger_word_pos_data)
	loader = data.DataLoader(dataset_, batch_size=128, collate_fn=collate_fn, shuffle=True)
	for i in tqdm(loader):
		pass
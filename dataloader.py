from torch.utils import data
from torch.nn.utils import rnn
import torch
from gensim.models import Word2Vec
import pickle
import tokenizer as tk
from tqdm import tqdm
from transformers import AlbertTokenizer


class dataset(data.Dataset):
    def __init__(
            self,
            tweet_pairs,
            distance_vectors,
            trigger_word_pos,
            common_words,
            day_difference,
            labels):
        self.tweet_pairs = tweet_pairs
        self.distance_vectors = distance_vectors
        self.vocab = None
        with open('./vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        self.trigger_word_pos = trigger_word_pos
        self.labels = labels
        self.common_words = common_words
        self.day_difference = day_difference

    def __len__(self):
        return len(self.tweet_pairs)

    def convert_tweet(self, tweet):
        return [self.vocab.stoi[word.strip()] for word in tk.tokenize(tweet)]

    def __getitem__(self, index):
        return torch.Tensor(self.convert_tweet(self.tweet_pairs[index][0])), torch.Tensor(self.convert_tweet(
            self.tweet_pairs[index][1])), torch.Tensor(self.distance_vectors[index][0]), torch.Tensor(
            self.distance_vectors[index][1]), torch.Tensor(self.trigger_word_pos[index][0]), torch.Tensor(
            self.trigger_word_pos[index][1]), torch.Tensor([self.labels[index]]), torch.Tensor(
            [self.common_words[index]]), torch.Tensor([self.day_difference[index]]), self.tweet_pairs[index][0], self.tweet_pairs[index][1]


def collate_fn(batch):
    alb_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    tweet1s = []
    tweet2s = []
    distance1s = []
    distance2s = []
    pos1s = []
    pos2s = []
    labels = []
    day_difference = []
    common_words = []
    # alb_tokens_1s = []
    # alb_tokens_2s = []
    tweet1 = []
    tweet2 = []
    for data in batch:
        tweet1s.append(data[0])
        tweet2s.append(data[1])
        distance1s.append(data[2])
        distance2s.append(data[3])
        pos1s.append(data[4].tolist())
        pos2s.append(data[5].tolist())
        labels.append(data[6].item())
        common_words.append(data[7].item())
        day_difference.append(data[8].item())
        tweet1.append(data[9])
        tweet2.append(data[10])

    alb_tokens_1s = alb_tokenizer(tweet1, return_tensors="pt", padding=True)
    alb_tokens_2s = alb_tokenizer(tweet2, return_tensors="pt", padding=True)
    packed_tweet1s = rnn.pack_sequence(tweet1s, enforce_sorted=False)
    packed_tweet2s = rnn.pack_sequence(tweet2s, enforce_sorted=False)
    packed_distance1s = rnn.pack_sequence(distance1s, enforce_sorted=False)
    packed_distance2s = rnn.pack_sequence(distance2s, enforce_sorted=False)
    alb_token_1s_input_ids = alb_tokens_1s['input_ids']
    alb_token_1s_token_type_ids = alb_tokens_1s['token_type_ids']
    alb_token_1s_attention_mask = alb_tokens_1s['attention_mask']
    alb_token_2s_input_ids = alb_tokens_2s['input_ids']
    alb_token_2s_token_type_ids = alb_tokens_2s['token_type_ids']
    alb_token_2s_attention_mask = alb_tokens_2s['attention_mask']
    padded_tweet1s, _ = rnn.pad_packed_sequence(
        packed_tweet1s, batch_first=True)
    padded_tweet2s, _ = rnn.pad_packed_sequence(
        packed_tweet2s, batch_first=True)
    padded_distance1s, _ = rnn.pad_packed_sequence(
        packed_distance1s, batch_first=True)
    padded_distance2s, _ = rnn.pad_packed_sequence(
        packed_distance2s, batch_first=True)
    return (
        padded_tweet1s,
        padded_tweet2s,
        padded_distance1s,
        padded_distance2s,
        torch.Tensor(pos1s),
        torch.Tensor(pos2s),
        torch.Tensor(common_words),
        torch.Tensor(day_difference),
        torch.Tensor(labels),
        alb_token_1s_input_ids,
        alb_token_1s_token_type_ids,
        alb_token_1s_attention_mask,
        alb_token_2s_input_ids,
        alb_token_2s_token_type_ids,
        alb_token_2s_attention_mask,
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
    trigger_word_pos_data = [[trigger_word_pos[i[0]],
                              trigger_word_pos[i[1]]] for i in tweet_pairs]
    common_words_data = [i[3] for i in tweet_pairs]
    day_difference_data = [i[4] for i in tweet_pairs]
    labels_data = [i[2] for i in tweet_pairs]

    dataset_ = dataset(
        tweet_pair_data, distance_vector_data,
        trigger_word_pos_data, common_words_data,
        day_difference_data, labels_data
    )
    loader = data.DataLoader(
        dataset_,
        batch_size=128,
        collate_fn=collate_fn,
        shuffle=True)
    for i in tqdm(loader):
        print(i)
        break

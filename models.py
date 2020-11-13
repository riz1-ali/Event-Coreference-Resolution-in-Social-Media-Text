import torch
from torch import nn
import pickle
from torch.autograd import Variable
from torchnlp.nn import Attention


class base_model(nn.Module):
    def __init__(
            self,
            device,
            max_dist,
            hidden_size=64,
            modelType="LSTM",
            distance_embedding_size=14):
        """
        Initializes the base_model class which is used for obtaining features of each tweet

        This function initializes the different modules of the pipeline
            - Loading word and distance embeddings
            - Initializing the bidirectional LSTM/GRU
            - Initializing the selective expression module
            - Initializing the attention layer

        Parameters:
            device                  (CPU/GPU)    : Device on which the model is being trained
            max_dist                (int)        : Maximum value in distance
            hidden_size             (int)        : Size of hidden outputs of LSTM/GRU
            modelType               (str)        : Type of RNN to use i.e. GRU/LSTM
            distance_embedding_size (int)        : Size of distance embeddings
        """
        super().__init__()
        self.hidden_size = hidden_size
        with open("./vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        pre_trained_emb = torch.Tensor(self.vocab.vectors)
        self.word_embedding = nn.Embedding.from_pretrained(pre_trained_emb)

        self.distance_embedding = nn.Embedding(
            max_dist, distance_embedding_size)

        if modelType == "LSTM":
            self.model = nn.LSTM(
                300 + distance_embedding_size,
                hidden_size,
                batch_first=True,
                bidirectional=True)
        if modelType == "GRU":
            self.model = nn.GRU(
                300 + distance_embedding_size,
                hidden_size,
                batch_first=True,
                bidirectional=True)
        self.bidirectional = True
        self.modelType = modelType

        if self.bidirectional:
            self.selective = nn.Linear(2 * hidden_size, 1)
            self.attention = Attention(2 * hidden_size)
        else:
            self.selective = nn.Linear(hidden_size, 1)
            self.attention = Attention(hidden_size)
        self.device = device

    def init_hidden(self, batch_size):
        if self.bidirectional:
            if self.modelType == "LSTM":
                h, c = (Variable(torch.zeros(1 * 2, batch_size, self.hidden_size)).to(self.device),
                        Variable(torch.zeros(1 * 2, batch_size, self.hidden_size)).to(self.device))
                return h, c
            elif self.modelType == "GRU":
                h = Variable(
                    torch.zeros(
                        1 * 2,
                        batch_size,
                        self.hidden_size)).to(
                    self.device)
                return h
        else:
            if self.modelType == "LSTM":
                h, c = (
                    Variable(
                        torch.zeros(
                            1, batch_size, self.hidden_size)).to(
                        self.device), Variable(
                        torch.zeros(
                            1, batch_size, self.hidden_size)).to(
                            self.device))
                return h, c
            elif self.modelType == "GRU":
                h = Variable(
                    torch.zeros(
                        1,
                        batch_size,
                        self.hidden_size)).to(
                    self.device)
                return h

    def forward(self, tweet, dist, pos):
        batch_size = tweet.shape[0]
        seq_len = tweet.shape[1]
        if self.modelType == "LSTM":
            h_0, c_0 = self.init_hidden(batch_size)
        elif self.modelType == "GRU":
            h_0 = self.init_hidden(batch_size)

        tweet_embedding = self.word_embedding(tweet.long())
        dist_embedding = self.distance_embedding(dist.long())

        tweet = torch.cat([tweet_embedding, dist_embedding], dim=2)
        if self.modelType == "LSTM":
            output, (h_n, c_n) = self.model(tweet, (h_0, c_0))
        elif self.modelType == "GRU":
            output, h_n = self.model(tweet, h_0)
        output_ = output.view(batch_size, seq_len, 2, self.hidden_size)

        indices = torch.Tensor(list(range(batch_size))).long()

        ment_part1 = output_[indices, pos[:, 1].long(), 0, :]
        ment_part2 = output_[indices, 0, 1, :]

        mention_feature = torch.cat([ment_part1, ment_part2], dim=1)

        Rc = output * mention_feature.view(batch_size, 1, -1)

        alpha = torch.tanh(self.selective(Rc))

        select = alpha * output

        t = self.attention(mention_feature.view(batch_size, 1, -1), select)

        Vem = torch.cat([t[0].view(batch_size, -1), mention_feature], dim=1)

        return Vem


class Model(nn.Module):
    def __init__(
            self,
            device,
            max_dist,
            hidden_size=64,
            modelType="LSTM",
            distance_embedding_size=14):
        """
        Initializes the overall model class which uses two base_model objects for each of the 2 tweets and then finally performs classification on the concatenated features.

        Parameters:
            device                  (CPU/GPU)    : Device on which the model is being trained
            max_dist                (int)        : Maximum value in distance
            hidden_size             (int)        : Size of hidden outputs of LSTM/GRU
            modelType               (str)        : Type of RNN to use i.e. GRU/LSTM
            distance_embedding_size (int)        : Size of distance embeddings

        Returns:
            A value corresponding to each element in the batch on which sigmoid is then applied to get the probability of the sample being in that class
        """
        super().__init__()
        with open("./vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        self.tweet1_model = base_model(
            device,
            max_dist,
            hidden_size,
            modelType,
            distance_embedding_size)
        self.tweet2_model = base_model(
            device,
            max_dist,
            hidden_size,
            modelType,
            distance_embedding_size)
        self.ds = nn.Linear(8 * hidden_size + 2, 64)
        self.final = nn.Linear(64, 1)

    def forward(
            self,
            tweet1,
            tweet2,
            dist1,
            dist2,
            pos1,
            pos2,
            common_words,
            day_difference):
        Vem1 = self.tweet1_model(tweet1, dist1, pos1)
        Vem2 = self.tweet2_model(tweet2, dist2, pos2)
        common_words = common_words.unsqueeze(1)
        day_difference = day_difference.unsqueeze(1)

        final = torch.cat([Vem1, Vem2, common_words, day_difference], dim=1)
        final = torch.relu(self.ds(final))
        return self.final(final)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from dataloader import dataset, collate_fn
import pickle
from torch.utils import data
from torch.autograd import Variable
from torchnlp.nn import Attention
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report


# ## Set Device for Training

# In[2]:


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# ## Load Data

# In[3]:


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

indices = list(range(len(tweet_pairs)))


# ## Train Test and Validation Split

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    indices, indices, test_size=0.6, random_state=42)


# In[5]:


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)


# In[6]:


X_train = [tweet_pairs[i] for i in X_train]
X_val = [tweet_pairs[i] for i in X_val]
X_test = [tweet_pairs[i] for i in X_test]


# In[7]:


tweet_pair_data_train = [[data_[i[0]], data_[i[1]]] for i in X_train]
distance_vector_data_train = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_train]
trigger_word_pos_data_train = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_train]
labels_data_train = [i[2] for i in X_train]
common_words_data_train = [i[3] for i in X_train]
day_difference_data_train = [i[4] for i in X_train]


# In[8]:


tweet_pair_data_val = [[data_[i[0]], data_[i[1]]] for i in X_val]
distance_vector_data_val = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_val]
trigger_word_pos_data_val = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_val]
labels_data_val = [i[2] for i in X_val]
common_words_data_val = [i[3] for i in X_val]
day_difference_data_val = [i[4] for i in X_val]


# In[9]:


tweet_pair_data_test = [[data_[i[0]], data_[i[1]]] for i in X_test]
distance_vector_data_test = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_test]
trigger_word_pos_data_test = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_test]
labels_data_test = [i[2] for i in X_test]
common_words_data_test = [i[3] for i in X_test]
day_difference_data_test = [i[4] for i in X_test]


# ## Set up Dataloader for train, test and validation splits

# In[10]:


dataset_ = dataset(
    tweet_pair_data_train, distance_vector_data_train,
    trigger_word_pos_data_train, common_words_data_train,
    day_difference_data_train, labels_data_train
)
loader_train = data.DataLoader(
    dataset_,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True)


# In[11]:


dataset_ = dataset(
    tweet_pair_data_val, distance_vector_data_val,
    trigger_word_pos_data_val, common_words_data_val,
    day_difference_data_val, labels_data_val
)
loader_val = data.DataLoader(
    dataset_,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True)


# In[12]:


dataset_ = dataset(
    tweet_pair_data_test, distance_vector_data_test,
    trigger_word_pos_data_test, common_words_data_test,
    day_difference_data_test, labels_data_test
)
loader_test = data.DataLoader(
    dataset_,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True)


# In[13]:


max_dist = -1

for i in distance_vectors:
    max_dist = max(max_dist, max(i))

max_dist += 1


# ## Base Model class for each tweet

# In[14]:


class base_model(nn.Module):
    def __init__(self, max_dist, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        with open("./vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        pre_trained_emb = torch.Tensor(self.vocab.vectors)
        self.word_embedding = nn.Embedding.from_pretrained(pre_trained_emb)

        self.distance_embedding = nn.Embedding(max_dist, 14)

        self.lstm = nn.LSTM(
            114,
            hidden_size,
            batch_first=True,
            bidirectional=True)

        self.selective = nn.Linear(2 * hidden_size, 1)

        self.attention = Attention(2 * hidden_size)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(1 * 2, batch_size, self.hidden_size)).to(device),
                Variable(torch.zeros(1 * 2, batch_size, self.hidden_size)).to(device))
        return h, c

    def forward(self, tweet, dist, pos):
        batch_size = tweet.shape[0]
        seq_len = tweet.shape[1]
        h_0, c_0 = self.init_hidden(batch_size)

        tweet_embedding = self.word_embedding(tweet.long())
        dist_embedding = self.distance_embedding(dist.long())

        tweet = torch.cat([tweet_embedding, dist_embedding], dim=2)
        output, (h_n, c_n) = self.lstm(tweet, (h_0, c_0))
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


# ## Class encapsulating model for each tweet and using output for final prediction

# In[15]:


class Model(nn.Module):
    def __init__(self, max_dist, hidden_size=64):
        super().__init__()
        with open("./vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        self.tweet1_model = base_model(max_dist, hidden_size)
        self.tweet2_model = base_model(max_dist, hidden_size)
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


# ## Model and Optimizer Initalization

# In[16]:


model = Model(max_dist).to(device)
optimizer = Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()


# ## Training and Validation Logic

# In[17]:


def train(model, optimizer, criterion, loader):
    losses = []
    model.train()
    for tweet1, tweet2, dist1, dist2, pos1, pos2, common_words, day_difference, label in tqdm(
            loader):
        optimizer.zero_grad()

        tweet1 = tweet1.to(device)
        tweet2 = tweet2.to(device)
        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        pos1 = pos1.to(device)
        pos2 = pos2.to(device)
        common_words = common_words.to(device)
        day_difference = day_difference.to(device)
        label = label.to(device)

        prediction = model(
            tweet1, tweet2,
            dist1, dist2,
            pos1, pos2,
            common_words, day_difference
        )

        loss = criterion(prediction.squeeze(), label.squeeze())

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


# In[18]:


def validate(model, criterion, loader):
    losses = []
    all_predictions, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for tweet1, tweet2, dist1, dist2, pos1, pos2, common_words, day_difference, label in tqdm(
                loader):

            tweet1 = tweet1.to(device)
            tweet2 = tweet2.to(device)
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            common_words = common_words.to(device)
            day_difference = day_difference.to(device)
            label = label.to(device)

            prediction = model(
                tweet1, tweet2,
                dist1, dist2,
                pos1, pos2,
                common_words, day_difference
            )

            loss = criterion(prediction.squeeze(), label.squeeze())
            all_predictions.extend(
                (prediction >= 0.5).long().squeeze().cpu().numpy().tolist())
            all_labels.extend(label.squeeze().cpu().numpy().tolist())
            losses.append(loss.item())
    return np.mean(losses), classification_report(all_labels, all_predictions)


# In[19]:


NUM_EPOCHS = 50
val_loss_store = 1e5


# In[ ]:


for epoch in range(NUM_EPOCHS):
    training_loss = train(model, optimizer, criterion, loader_train)
    val_loss, val_report = validate(model, criterion, loader_val)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} Training Loss : {training_loss} Validation Loss : {val_loss}")
    print(val_report)
    print("----------------------------------------------------------------------")

    if val_loss_store > val_loss:
        val_loss_store = val_loss
        torch.save(model.state_dict(), 'model.tar')


# In[ ]:

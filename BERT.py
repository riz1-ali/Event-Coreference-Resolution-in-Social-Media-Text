#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from ray.util.sgd.torch import TrainingOperator


# In[2]:


import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils import data
import sys
from tqdm import tqdm
from dataloader import collate_fn, dataset
from models import Model
from utils import train, validate
import pickle
from sklearn.model_selection import train_test_split


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
max_dist = -1

for i in distance_vectors:
    max_dist = max(max_dist, max(i))

max_dist += 1


# In[4]:


device = torch.device('cuda:0')


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    indices, indices, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

X_train = [tweet_pairs[i] for i in X_train]
X_val = [tweet_pairs[i] for i in X_val]
X_test = [tweet_pairs[i] for i in X_test]


tweet_pair_data_train = [[data_[i[0]], data_[i[1]]] for i in X_train]
distance_vector_data_train = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_train]
trigger_word_pos_data_train = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_train]
labels_data_train = [i[2] for i in X_train]
common_words_data_train = [i[3] for i in X_train]
day_difference_data_train = [i[4] for i in X_train]


tweet_pair_data_val = [[data_[i[0]], data_[i[1]]] for i in X_val]
distance_vector_data_val = [
    [distance_vectors[i[0]], distance_vectors[i[1]]] for i in X_val]
trigger_word_pos_data_val = [
    [trigger_word_pos[i[0]], trigger_word_pos[i[1]]] for i in X_val]
labels_data_val = [i[2] for i in X_val]
common_words_data_val = [i[3] for i in X_val]
day_difference_data_val = [i[4] for i in X_val]


# In[6]:


class ModOperator(TrainingOperator):
    def setup(self, args):
        train_dataset = dataset(
            tweet_pair_data_train, distance_vector_data_train, 
            trigger_word_pos_data_train, common_words_data_train,
            day_difference_data_train, labels_data_train
        )
        val_dataset = dataset(
            tweet_pair_data_val, distance_vector_data_val, 
            trigger_word_pos_data_val, common_words_data_val,
            day_difference_data_val, labels_data_val
        )
        train_loader = data.DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
        
        model = Model(device, max_dist, modelType="LSTM", hidden_size=128)
        model.load_state_dict(torch.load("./model_BERT.tar"))
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        self.model, self.optimizer, self.criterion = self.register(
            models=model, 
            optimizers=optimizer,
            criterion=criterion,
            schedulers=None
        )
        self.register_data(train_loader=train_loader, validation_loader=val_loader)
        self.VAL = 0
    
    def train_batch(self, batch, batch_info):
        tweet1, tweet2, dist1, dist2, pos1, pos2, common_words, day_difference, label, alb_token_1s_input_ids, alb_token_1s_token_type_ids, alb_token_1s_attention_mask, alb_token_2s_input_ids, alb_token_2s_token_type_ids, alb_token_2s_attention_mask = batch
        model = self.model
        optimizer = self.optimizer
        tweet1 = tweet1.cuda()
        tweet2 = tweet2.cuda()
        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        common_words = common_words.cuda()
        day_difference = day_difference.cuda()
        label = label.cuda()
        alb_token_1s_input_ids = alb_token_1s_input_ids.cuda()
        alb_token_1s_token_type_ids = alb_token_1s_token_type_ids.cuda()
        alb_token_1s_attention_mask = alb_token_1s_attention_mask.cuda()
        alb_token_2s_input_ids = alb_token_2s_input_ids.cuda()
        alb_token_2s_token_type_ids = alb_token_2s_token_type_ids.cuda()
        alb_token_2s_attention_mask = alb_token_2s_attention_mask.cuda()
        
        optimizer.zero_grad()
        prediction = model(
            tweet1, tweet2,
            dist1, dist2,
            pos1, pos2,
            common_words, day_difference, 
            alb_token_1s_input_ids, alb_token_1s_token_type_ids, 
            alb_token_1s_attention_mask, alb_token_2s_input_ids, 
            alb_token_2s_token_type_ids, alb_token_2s_attention_mask
        )

        loss = self.criterion(prediction.squeeze(), label.squeeze())
        loss.backward()
        optimizer.step()
        
        return {
            'loss' : loss.item()
        }
    
    def validate(self, val_iterator, info):
        self.model.eval()
        losses = []
        all_predictions, all_labels = [], []
        for batch in val_iterator:
            vals = self.validate_batch(batch, None)
            prediction = vals['predictions']
            label = vals['labels']
            loss = vals['loss']
            pred = (prediction >= 0.5)
            if len(pred.shape) == 0:
                all_predictions.append(pred.tolist())
            else:
                all_predictions.extend(pred.tolist())
            if len(label.shape) == 0:
                all_labels.append(label.tolist())
            else:
                all_labels.extend(label.tolist())
            losses.append(loss)
        report = classification_report(all_labels, all_predictions, output_dict=True)
        report['loss'] = np.mean(losses)
        return report
        
    
    def validate_batch(self, batch, batch_info):
        tweet1, tweet2, dist1, dist2, pos1, pos2, common_words, day_difference, label, alb_token_1s_input_ids, alb_token_1s_token_type_ids, alb_token_1s_attention_mask, alb_token_2s_input_ids, alb_token_2s_token_type_ids, alb_token_2s_attention_mask = batch
        model = self.model
        optimizer = self.optimizer
        tweet1 = tweet1.cuda()
        tweet2 = tweet2.cuda()
        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        common_words = common_words.cuda()
        day_difference = day_difference.cuda()
        label = label.cuda()
        alb_token_1s_input_ids = alb_token_1s_input_ids.cuda()
        alb_token_1s_token_type_ids = alb_token_1s_token_type_ids.cuda()
        alb_token_1s_attention_mask = alb_token_1s_attention_mask.cuda()
        alb_token_2s_input_ids = alb_token_2s_input_ids.cuda()
        alb_token_2s_token_type_ids = alb_token_2s_token_type_ids.cuda()
        alb_token_2s_attention_mask = alb_token_2s_attention_mask.cuda()
        prediction = model(
            tweet1, tweet2,
            dist1, dist2,
            pos1, pos2,
            common_words, day_difference, 
            alb_token_1s_input_ids, alb_token_1s_token_type_ids, 
            alb_token_1s_attention_mask, alb_token_2s_input_ids, 
            alb_token_2s_token_type_ids, alb_token_2s_attention_mask
        )

        loss = self.criterion(prediction.squeeze(), label.squeeze())
        
        return {
            'loss' : loss.item(),
            'predictions' : prediction.detach().squeeze().cpu().numpy(),
            'labels' : label.detach().squeeze().cpu().numpy(),
        }


# In[7]:


import ray
ray.shutdown()


# In[8]:


ray.init()


# In[9]:


from ray.util.sgd import TorchTrainer


# In[10]:


trainer = TorchTrainer(
    training_operator_cls=ModOperator,
    config={"lr": 0.001},
    num_workers=4,
    scheduler_step_freq="epoch",
    use_gpu=True,
    use_tqdm=False
)


# In[11]:


val_dict = {}
val_loss_store = 1e5
for epoch in tqdm(range(10)):
    losses = trainer.train()
    val_report = trainer.validate()
    val_dict[epoch] = val_report
    with open("val_results_BERT.pkl", "wb") as f:
        pickle.dump(val_dict, f)
    val_loss = val_report['loss']
    if val_loss < val_loss_store:
        val_loss_store = val_loss
        trainer.save("./BERT_trainer")
        torch.save(trainer.get_model().state_dict(), "model_BERT_update.tar")
    print(f"{epoch + 1}/{10} : Training Loss {losses['loss']} Validation Loss {val_loss}")


# In[ ]:





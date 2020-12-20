#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models import Model
import torch
from utils import validate
import pickle
from torch import nn


# In[2]:


with open('./distance_vectors.pkl', 'rb') as f:
    distance_vectors = pickle.load(f)
max_dist = -1

for i in distance_vectors:
    max_dist = max(max_dist, max(i))

max_dist += 1


# In[3]:


with open("test_loader.pkl", "rb") as f:
    loader_val = pickle.load(f)


# In[4]:


device = torch.device("cuda:0")


# In[20]:


model = Model(device, max_dist, 128, "LSTM").to(device)


# In[21]:


model.load_state_dict(torch.load('./model_LSTM_0.01_128_Scheduler(True).tar'))


# In[22]:


criterion = nn.BCEWithLogitsLoss()


# In[23]:


rep = validate(model, criterion, loader_val, device)


# In[24]:


print(rep)


# In[ ]:





import argparse
import pickle

import numpy as np
import torch
import wandb
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils import data
import sys
from tqdm import tqdm

from dataloader import collate_fn, dataset
from models import Model
from utils import train, validate

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modelType",
    help="Type of Model LSTM/GRU",
    default="LSTM")
parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-2)
parser.add_argument("--device", help="CPU/GPU", default="GPU")
parser.add_argument(
    "--hidden_size",
    help="Size of hidden state of RNN",
    default=64,
    type=int)
parser.add_argument(
    "--num_epochs",
    help="Number of epochs",
    type=int,
    default=50)
parser.add_argument(
    "--use_scheduler",
    help="Use Scheduler",
    type=bool,
    default=False)
parser.add_argument(
    "--use_wandb",
    help="Logging using wandb",
    type=bool,
    default=False)
args = parser.parse_args()

# Set Device for training
if args.device == "CPU":
    device = torch.device('cpu')
elif args.device == "GPU":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print("Sorry! GPU not available Using CPU instead")
        device = torch.device('cpu')
else:
    print("Invalid Device")
    quit(0)

# Initialize wandb for logging
if args.use_wandb:
    wandb.init(project='EventCoreference')
    wandb.config.update(args)

# Load dataloader for training and testing
with open("train_loader.pkl", "rb") as f:
    loader_train = pickle.load(f)

with open("val_loader.pkl", "rb") as f:
    loader_val = pickle.load(f)

with open('./distance_vectors.pkl', 'rb') as f:
    distance_vectors = pickle.load(f)
max_dist = -1

for i in distance_vectors:
    max_dist = max(max_dist, max(i))

max_dist += 1

# Initializing model with hyperparameters
model = Model(device, max_dist, modelType=args.modelType,
              hidden_size=args.hidden_size).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
if args.use_scheduler:
    print("Initializing Scheduler")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
criterion = nn.BCEWithLogitsLoss()

val_loss_store = 1e5

# Training and validation
for epoch in range(args.num_epochs):
    training_loss = train(model, optimizer, criterion, loader_train, device)
    val_loss, val_report = validate(model, criterion, loader_val, device)
    if val_loss < val_loss_store:
        val_loss_store = val_loss
        torch.save(model.state_dict(), f"fasttext_model_{args.modelType}_{args.lr}_{args.hidden_size}_Scheduler({args.use_scheduler}).tar")
    if args.use_scheduler:
        scheduler.step(val_loss)
    if args.use_wandb:
        wandb.log({
            "training loss": training_loss,
            "Validation loss": val_loss,
            "Positive F1": val_report['1.0']['f1-score'],
            "Positive Precision": val_report['1.0']['precision'],
            "Positive Recall": val_report['1.0']['recall'],
            "Positive Support": val_report['1.0']['support'],
            "Negative F1": val_report['0.0']['f1-score'],
            "Negative Precision": val_report['0.0']['precision'],
            "Negative Recall": val_report['0.0']['recall'],
            "Negative Support": val_report['0.0']['support'],
        })
    print(f"{epoch + 1}/{args.num_epochs} : Training Loss {training_loss} Validation Loss {val_loss}")
    print(val_report)
    sys.stdout.flush()

import torch
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm


def train(model, optimizer, criterion, loader, device):
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


def validate(model, criterion, loader, device):
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
    return np.mean(losses), classification_report(
        all_labels, all_predictions, output_dict=True)

# model_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

def train_epoch(model, dl, opt):
    for xb, yb in dl:
        preds = model(xb)
        loss = mnist_loss(preds, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

def validate_epoch(model, valid_dl):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

def train_model(model, epochs, dl, valid_dl, lr):
    opt = SGD(model.parameters(), lr)
    for i in range(epochs):
        train_epoch(model, dl, opt)
        print(validate_epoch(model, valid_dl), end=' ')
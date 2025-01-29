# utils.py
import torch
from torch import tensor

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

def mnist_distance(a, b):
    return (a - b).abs().mean((-1, -2))

def is_3(x, mean3, mean7):
    return mnist_distance(x, mean3) < mnist_distance(x, mean7)
# advanced_models.py
from fastai.vision.all import *

def create_cnn_learner(dls):
    learn = cnn_learner(dls, resnet18, pretrained=False, loss_func=F.cross_entropy, metrics=accuracy)
    return learn

def train_cnn(learn, epochs, lr):
    learn.fit_one_cycle(epochs, lr)

def simple_net():
    return nn.Sequential(
        nn.Linear(28*28, 30),
        nn.ReLU(),
        nn.Linear(30, 1)
    )
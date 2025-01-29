# main.py
from data_preparation import prepare_data, load_image_paths, load_tensors
from data_visualization import show_image, plot_function
from model_training import mnist_loss, batch_accuracy, train_epoch, validate_epoch, train_model
from advanced_models import simple_net, create_cnn_learner, train_cnn
from optimization import BasicOptim
import torch
from torch import nn
from fastai.vision.all import *

# Prepare data
path = prepare_data()
threes, sevens, valid_threes, valid_sevens = load_image_paths(path)

# Load tensors
three_tensors = load_tensors(threes)
seven_tensors = load_tensors(sevens)
valid_3_tens = load_tensors(valid_threes)
valid_7_tens = load_tensors(valid_sevens)

# Stack tensors
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_sevens = torch.stack(seven_tensors).float() / 255
valid_3_tens = torch.stack(valid_3_tens).float() / 255
valid_7_tens = torch.stack(valid_7_tens).float() / 255

# Prepare DataLoaders
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
dset = list(zip(train_x, train_y))
dl = DataLoader(dset, batch_size=256)

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))
valid_dl = DataLoader(valid_dset, batch_size=256)

# Train linear model
linear_model = nn.Linear(28*28, 1)
train_model(linear_model, 20, dl, valid_dl, lr=1.0)

# Train CNN model
dls = ImageDataLoaders.from_folder(path)
learn = create_cnn_learner(dls)
train_cnn(learn, 1, 0.1)
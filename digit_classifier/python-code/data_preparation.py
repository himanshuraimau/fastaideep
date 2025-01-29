# data_preparation.py
from fastai.vision.all import *
from fastbook import *

def prepare_data():
    path = untar_data(URLs.MNIST_SAMPLE)
    Path.BASE_PATH = path
    return path

def load_image_paths(path):
    threes = (path/'train'/'3').ls().sorted()
    sevens = (path/'train'/'7').ls().sorted()
    valid_threes = (path/'valid'/'3').ls().sorted()
    valid_sevens = (path/'valid'/'7').ls().sorted()
    return threes, sevens, valid_threes, valid_sevens

def load_tensors(image_paths):
    return [tensor(Image.open(o)) for o in image_paths]
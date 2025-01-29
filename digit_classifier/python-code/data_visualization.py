# data_visualization.py
import matplotlib.pyplot as plt
from fastai.vision.all import *

def show_image(tensor):
    plt.imshow(tensor, cmap='Greys')
    plt.show()

def plot_function(f, title=None, min=-4, max=4):
    x = torch.linspace(min, max, 100)
    plt.plot(x, f(x))
    if title:
        plt.title(title)
    plt.show()
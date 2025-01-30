import numpy as np
import matplotlib.pyplot as plt

def plot_function(f, min=-2.1, max=2.1, color='r'):
    x = np.linspace(min, max, 100)[:, None]
    plt.plot(x, f(x), color)

def show_corr(df, a, b, corr_func):
    x, y = df[a], df[b]
    plt.scatter(x, y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr_func(x, y):.2f}')

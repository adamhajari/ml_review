import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_training_data(n, m=-1, b=0):
    x0 = np.random.rand(n)*10 - 5
    x1 = np.random.rand(n)*10 - 5
    # x0 = -x1*(w1/w2) - (b/w2)
    labels = [1 if xx0 + m*xx1 + b < 0 else 0 for xx0, xx1 in zip(x0, x1)]
    train_data = pd.DataFrame({'x0': x0, 'x1': x1, 'label': labels})
    return train_data

def plot_data(data):
    groups = data.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x0, group.x1, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    plt.show()
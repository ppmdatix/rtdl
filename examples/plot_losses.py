import numpy as np
from matplotlib import pyplot as plt

def plot_losses(_losses, title="this is a graph", path=None):
    for key in _losses:
        plt.plot([np.log(x) for x in _losses[key]], label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(title)
    plt.show()
    if path is not None:
        plt.savefig(path)
    plt.close()
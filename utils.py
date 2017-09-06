import matplotlib.pyplot as plt
import numpy as np

def plot_running_average(total_rewards, title="Running average", filename="fig.png", bin=10, show=False):
    N = len(total_rewards)
    running_average = np.empty(N)
    for t in range(N):
        running_average[t] = total_rewards[max(0, t-bin):t-1].mean()

    plt.figure()
    plt.plot(running_average)
    plt.title(title)
    if show:
        plt.show()
    plt.savefig(filename)
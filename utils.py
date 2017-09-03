import matplotlib.pyplot as plt
import numpy as np

def plot_running_average(total_rewards, bin=10):
    N = len(total_rewards)
    running_average = np.empty(N)
    for t in range(N):
        running_average[t] = total_rewards[max(0, t-bin):t-1].mean()

    plt.plot(running_average)
    plt.title("Running average")
    plt.show()
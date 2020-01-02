import numpy as np
import matplotlib.pyplot as plt


def get_hellinger_distance(h1, h2):
    var_tmp = 0
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    for h1_value, h2_value in zip(h1, h2):
        var_tmp += np.sqrt(h1_value * h2_value)
    distance = np.sqrt(1 - var_tmp)
    return distance


def main():
    rng = np.random.RandomState(10)  # deterministic random data
    mean_shifts = [0,4,10,20,70]
    mean = 128
    for i in range(len(mean_shifts)):
        data = rng.normal(loc=mean, scale=10, size=1000)
        data_2 = rng.normal(loc=mean - mean_shifts[i], scale=10, size=1000)
        hist_data, hist_bins = np.histogram(data, bins=50, range=(0,256))
        hist_data_2, hist_bins_2 = np.histogram(data_2, bins=50, range=(0,256))
        distance_hellinger = get_hellinger_distance(hist_data, hist_data_2)
        plt.subplot(len(mean_shifts), 1, i + 1)
        n_bar, bins, patches = plt.hist(data, bins=50, range=(0,256), label='1. Histogram')
        n_bar_2, bins_2, patches_2 = plt.hist(data_2, bins=50, range=(0,256),color='red',label='2. Histogram')
        plt.title('Hellinger distance: {0:0.2f} with mean shift {1}'.format(distance_hellinger, mean_shifts[i]))
        plt.legend()
        if i < len(mean_shifts) - 1:
            plt.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
    plt.show()


if __name__ == '__main__':
    main()

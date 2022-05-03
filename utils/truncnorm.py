from scipy.stats import truncnorm
import numpy as np

def get_truncated_normal(mean=0, sd=1, low=0, high=10, shape=(1), mask=[]):
    #print(shape, mask.shape)
    zero_array = np.zeros(shape)

    X = truncnorm(
        (low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)

    for i in range(zero_array.shape[0]):
        for j in range(zero_array.shape[1]):
            if mask[i][j] == 1:
                zero_array[i][j] = X.rvs(1)

    

    return zero_array
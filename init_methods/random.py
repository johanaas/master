from utils import decision_function
import numpy as np
from matplotlib import pyplot as plt
import copy
import query_counter

def get_random_noise(model, params):
    num_evals = 0
    success = 0
    while True:
        random_noise = np.random.uniform(params['clip_min'], 
            params['clip_max'], size = params['shape'])
        success = decision_function(model,random_noise[None], params)[0]
        query_counter.queries = 0
        num_evals += 1
        if success:
            break
        assert num_evals < 1e4,"Initialization failed! "
    return random_noise
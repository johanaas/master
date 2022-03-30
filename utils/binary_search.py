import numpy as np

from utils.decision_function import decision_function

def binary_search(model, sample, noise, params):
    
    blended = np.copy(sample)

    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        blended = (1 - mid) * sample + mid * noise 
        success = decision_function(model, blended[None], params)
        if success:
            high = mid
        else:
            low = mid

    final_img = (1 - high) * sample + high * noise
    return final_img
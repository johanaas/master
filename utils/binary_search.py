import numpy as np

from utils.decision_function import decision_function
from utils.clip_image import clip_image

def binary_search(model, sample, noise, params):
    
    blended = np.copy(sample)

    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        blended = (1 - mid) * sample + mid * noise 
        blended = clip_image(blended, params['clip_min'], params['clip_max'])
        success = decision_function(model, blended[None], params, sample)
        if success:
            high = mid
        else:
            low = mid

    final_img = (1 - high) * sample + high * noise
    return final_img
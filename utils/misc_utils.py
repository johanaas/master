import numpy as np
import copy

def compute_distance(x_ori, x_pert, constraint = 'l2'):
	# Compute the distance between two images.
	if constraint == 'l2':
		return np.linalg.norm(x_ori - x_pert)
	elif constraint == 'linf':
		return np.max(abs(x_ori - x_pert))

def decision_function(model, images, params):
	"""
	Decision function output 1 on the desired side of the boundary,
	0 otherwise.
	"""
	images = clip_image(images, params['clip_min'], params['clip_max'])
	prob = model.predict(images)
	if params['target_label'] is None:
		return np.argmax(prob, axis = 1) != params['original_label'] 
	else:
		return np.argmax(prob, axis = 1) == params['target_label']

def clip_image(image, clip_min, clip_max):
	# Clip an image, or an image batch, with upper and lower threshold.
	return np.minimum(np.maximum(clip_min, image), clip_max) 

def binary_search(model, sample, noise, params):
    
    blended = copy.deepcopy(sample)

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
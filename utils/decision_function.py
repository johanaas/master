import numpy as np

from utils.clip_image import clip_image
from defence.jpeg import runJPEG

def decision_function(model, images, params, org_img = []):
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

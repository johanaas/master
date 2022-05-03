import numpy as np
import query_counter
import evaluation
import config as CFG

from utils.clip_image import clip_image
from defence.jpeg import runJPEG

def decision_function(model, images, params, org_img = []):
	"""
	Decision function output 1 on the desired side of the boundary,
	0 otherwise.
	"""
	images = clip_image(images, params['clip_min'], params['clip_max'])
	
	# Run defence if enabled in config 
	if CFG.DEFENCE == "JPEG":
		images = runJPEG(images, org_img)
	
	#print("l2: ", np.linalg.norm(org_img - images[0]))
	
	prob = model.predict(images)

	if len(org_img) > 0:
		for i, img in enumerate(images):
			query_counter.queries += 1
			
			if len(images) > 1:
				evaluation.add_dist_queries(query_counter.prev_dist)
				continue

			if np.argmax(prob[i]) != params['original_label']:
				dist = np.linalg.norm(img - org_img)
				evaluation.add_dist_queries(dist)
				query_counter.prev_dist = dist
			else:
				# Use prev_dist
				evaluation.add_dist_queries(query_counter.prev_dist)

	if params['target_label'] is None:
		return np.argmax(prob, axis = 1) != params['original_label'] 
	else:
		return np.argmax(prob, axis = 1) == params['target_label']

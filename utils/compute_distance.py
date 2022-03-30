import numpy as np

def compute_distance(x_ori, x_pert, constraint = 'l2'):
	# Compute the distance between two images.
	if constraint == 'l2':
		return np.linalg.norm(x_ori - x_pert)
	elif constraint == 'linf':
		return np.max(abs(x_ori - x_pert))
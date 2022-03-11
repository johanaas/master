from .random import get_random_noise
from .par import get_par_patches
from .deconvolution import deconvolute

def get_start_image(
  experiment=None,
  sample=None,
  model=None,
  params=None):

  if experiment == "random":
    return get_random_noise(model, params)
  elif experiment == "par":
    return get_par_patches(sample, model, params)
  elif experiment == "deconvolute":
    return deconvolute(sample, get_par_patches(sample, model, params))
  else:
    raise ValueError("No init method with the name:", experiment)

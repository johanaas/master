from .random import get_random_noise
from .par import get_par_patches
from .fourier import get_fourier_perturbation
from .dct import get_dct_image

def get_start_image(
  experiment=None,
  sample=None,
  model=None,
  params=None):

  if experiment == "random":
    return get_random_noise(model, params)
  elif experiment == "par":
    return get_par_patches(sample, model, params)
  elif experiment == "fourier":
    return get_fourier_perturbation(sample, model, params)
  elif experiment == "dct":
    return get_dct_image(sample, model, params)
  else:
    raise ValueError("No init method with the name:", experiment)

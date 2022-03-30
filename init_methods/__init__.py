from .random import get_random_noise
from .par import get_par_patches
from .dct import get_dct_image
from .fourier import create_fperturb_binary_seach_in_different_freq
from .fpar import get_fourier_par

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
    return create_fperturb_binary_seach_in_different_freq(sample, model, params)
  elif experiment == "fpar":
    return get_fourier_par(sample, model, params)
  elif experiment == "dct":
    return get_dct_image(sample, model, params)
  else:
    raise ValueError("No init method with the name:", experiment)

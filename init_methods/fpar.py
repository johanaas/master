from .fourier import create_fperturb_binary_seach_in_different_freq
from .cfpar import create_fperturb_binary_seach_in_different_freq as cfunction
from .improved_fourier import fourier_attack
from .par import get_par_patches

import numpy as np
from utils.clip_image import clip_image

def get_fourier_par(sample, model, params):
    fourier_noise_on_img = create_fperturb_binary_seach_in_different_freq(np.copy(sample), model, params)
    fourier_noise_on_img = clip_image(np.copy(fourier_noise_on_img), clip_max=1.0, clip_min=0.0)
    fpar_img = get_par_patches(sample, model, params, noise=np.copy(fourier_noise_on_img), plot_each_step=False)

    return fpar_img

def improved_fpar(sample, model, params):
    fourier_noise_on_img = fourier_attack(np.copy(sample), model, params)
    fourier_noise_on_img = clip_image(np.copy(fourier_noise_on_img), clip_max=1.0, clip_min=0.0)
    fpar_img = get_par_patches(sample, model, params, noise=np.copy(fourier_noise_on_img), plot_each_step=False)

    return fpar_img

def circular_fpar(sample, model, params):
    fourier_noise_on_img = cfunction(np.copy(sample), model, params)
    fourier_noise_on_img = clip_image(np.copy(fourier_noise_on_img), clip_max=1.0, clip_min=0.0)
    fpar_img = get_par_patches(sample, model, params, noise=np.copy(fourier_noise_on_img), plot_each_step=False)

    return fpar_img
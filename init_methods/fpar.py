from .fourier import create_fperturb_binary_seach_in_different_freq
from .par import get_par_patches

import numpy as np
from utils import clip_image, compute_distance

from matplotlib import pyplot as plt

def get_fourier_par(sample, model, params):
    fourier_noise_on_img = create_fperturb_binary_seach_in_different_freq(np.copy(sample), model, params)
    fourier_noise_on_img = clip_image(np.copy(fourier_noise_on_img), clip_max=1.0, clip_min=0.0)
    fpar_img = get_par_patches(sample, model, params, noise=np.copy(fourier_noise_on_img), plot_each_step=False)

    """
    print()
    print()
    print("Distance before par:", compute_distance(sample, fourier_noise_on_img))
    print("Distance after par :", compute_distance(sample, fpar_img))
    print()
    print()

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(sample)
    axs[0].set_title("Original")
    axs[1].imshow(fourier_noise_on_img)
    axs[1].set_title("Fourier")
    axs[2].imshow(fpar_img)
    axs[2].set_title("Fourier + par")
    plt.show()
    """

    return fpar_img

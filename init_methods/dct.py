import numpy as np
from scipy.fftpack import dct, idct
import cv2
from utils import decision_function
from matplotlib import pyplot as plt

def get_dct_image(orig_img, model, params, band_width=50, step_size=40, perturbation_factor=1):

    y, u, v = get_yuv_from_rgb(orig_img)
    random_noise = np.random.uniform(0, 1, y.shape)

    dct_coeffs = dct2(y)

    coeffs_max = np.max(dct_coeffs)

    epsilon = 0.01

    for i in range(5):
        thresh = coeffs_max * epsilon**i
        mask = (np.abs(dct_coeffs) < thresh)

        perturbed_coeffs = np.copy(dct_coeffs)
        perturbed_coeffs[mask] = random_noise[mask]

        perturbed_img = get_rgb_from_yuv([idct2(perturbed_coeffs), u, v])

        fig, axs = plt.subplots(2, 1)

        axs[0].imshow(perturbed_img)
        axs[1].imshow(perturbed_coeffs)
        plt.show()

        if model.predict(perturbed_img): # TODO: change to actual model. True if adversarial
            return perturbed_img
        
        break

    print("\n\n\nImage is not adversarial. Throwing exception\n\n\n")
    raise ValueError("Could not create adversarial image")

    """
    # DCT is faster and easier in 2D, thus we use
    # the limunance component of the YUV format
    img, u, v = get_yuv_from_rgb(orig_img)

    dct_coeffs = dct2(img)

    masks_generator = get_diag_masks(img.shape[0], band_width=band_width, step_size=step_size)

    for mask in masks_generator:
        perturbed_coeffs = np.copy(dct_coeffs)

        # TODO: Use some other method to update the coeffs instead of numpy random uniform and multiplying
        perturbed_coeffs[mask] = perturbed_coeffs[mask] * np.random.uniform(-perturbation_factor, perturbation_factor, perturbed_coeffs[mask].shape)

        perturbed_y = idct2(perturbed_coeffs)
        perturbed_img = get_rgb_from_yuv(perturbed_y, u, v)

        if decision_function(model, perturbed_img[None], params)[0]:
            return perturbed_img

    print("Could not find adversarial image through DCT with step size band_width {} and step size {}.".format(band_width, step_size))
    return None
    """

def get_diag_masks(img_width, band_width=1, step_size=1):
    mask_container = np.ones((img_width, img_width))

    offset = 0
    while offset < (img_width * 2) - 2:
        band_img = np.copy(mask_container)
        band_img = np.rot90(band_img, 2) # Rotate to search from bottom right

        prev_row = None

        # Add zeros along band
        iterations = band_width + offset
        for i in range(iterations):
            true_start = offset - i
            start = np.maximum(0, np.minimum(band_img.shape[0], true_start))

            true_end = true_start + band_width
            end = np.minimum(band_img.shape[0], np.maximum(0, true_end))

            row = np.minimum(band_img.shape[0] - 1, i)

            if start == end or row == prev_row:
                continue

            prev_row = row
            band_img[row, start:end] = 0

        band_img = np.rot90(band_img, 2)

        offset += step_size

        mask = (band_img == 0)
        yield mask

def get_yuv_from_rgb(img):
    img_yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(img_yuv)
    return y, u, v

def get_rgb_from_yuv(channels):
    img_yuv = np.dstack(channels)
    img = cv2.cvtColor(np.float32(img_yuv), cv2.COLOR_YUV2RGB)
    return img

# dct2 and idct2 implementations from Sandipan Dey's answer here:
# https://stackoverflow.com/questions/7110899/how-do-i-apply-a-dct-to-an-image-in-python

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

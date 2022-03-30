import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from utils import decision_function, compute_distance
import cv2
import scipy.fftpack as fp
import copy
import jenkspy

def create_fperturb_binary_seach_in_different_freq(img, model, params):
    
    perturbed = create_fperurb_rgb(img, model, params)

    band_mask = np.ones((224,224))
    cv2.circle(band_mask, (112,112), 25, 0, -1)

    low_mask = (band_mask == 0)

    band_mask = np.ones((224,224))
    cv2.circle(band_mask, (112,112), 75, 0, -1)
    cv2.circle(band_mask, (112,112), 25, 1, -1)

    med_mask = (band_mask == 0)

    band_mask = np.ones((224,224))
    cv2.circle(band_mask, (112,112), 75, 0, -1)

    high_mask = (band_mask == 1)



    final_image1 = create_fperturb_binary_seach(low_mask, perturbed, img, model, params)

    final_image2 = create_fperturb_binary_seach(med_mask, final_image1, img, model, params)

    final_image3 = create_fperturb_binary_seach(high_mask, final_image2, img, model, params)

    return final_image3

def create_fperturb_binary_seach(mask, perturbed, img, model, params):
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        transformed_channels = []

        for i in range(3):
            rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, i])))
            magnitude = np.abs(rgb_fft)
            
            org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
            org_magnitude = np.abs(org_fft)
            blended = copy.deepcopy(magnitude)

            blended[mask] = (1 - mid) * org_magnitude[mask] + mid * magnitude[mask]
            phase = np.angle(rgb_fft, deg=False)
            b = blended*np.sin(phase)
            a = blended*np.cos(phase)

            z = a + b * 1j
            back_shift = np.fft.ifftshift(z)
            bb = np.fft.ifft2(back_shift).real

            transformed_channels.append(bb)

        final_image = np.dstack([transformed_channels[0].astype(float), 
                                transformed_channels[1].astype(float), 
                                transformed_channels[2].astype(float)])

        success = decision_function(model, final_image[None], params)

        if success:
            high = mid
        else:
            low = mid

    transformed_channels = []

    for i in range(3):
            rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, i])))
            magnitude = np.abs(rgb_fft)

            org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
            org_magnitude = np.abs(org_fft)
            blended = copy.deepcopy(magnitude)
            blended[mask] = (1 - high) * org_magnitude[mask] + high * magnitude[mask]
            phase = np.angle(rgb_fft, deg=False)
            b = blended*np.sin(phase)
            a = blended*np.cos(phase)

            z = a + b * 1j
            back_shift = np.fft.ifftshift(z)
            bb = np.fft.ifft2(back_shift).real

            transformed_channels.append(bb)

    
    final_image = np.dstack([transformed_channels[0].astype(float), 
                                transformed_channels[1].astype(float), 
                                transformed_channels[2].astype(float)])

    return final_image

def create_fperurb_rgb(img, model, params):

    transformed_channels = []
    band_mask = np.ones((224,224))
    cv2.circle(band_mask, (112,112), 56, 0, -1)
    mask = (band_mask != 0)
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
        
        random_noise = np.random.uniform(0, 1, size = (224,224))
        magnitude = np.log(np.abs(rgb_fft))
        
        high_mask = (magnitude < 1)
        med_mask = (magnitude > 2) | (magnitude < 4)
        low_mask = (magnitude > 5)

        magnitude[high_mask] = np.random.uniform(0, 1, size = (224,224))[high_mask]
        magnitude[med_mask] = np.random.uniform(2, 4, size = (224,224))[med_mask]
        magnitude[low_mask] = np.random.uniform(5, 9, size = (224,224))[low_mask]

        magnitude = np.exp(magnitude)
        phase = np.angle(rgb_fft, deg=False)
        b = magnitude*np.sin(phase)
        a = magnitude*np.cos(phase)

        z = a + b * 1j
        back_shift = np.fft.ifftshift(z)
        bb = np.fft.ifft2(back_shift).real

        transformed_channels.append(bb)
    
    final_image = np.dstack([transformed_channels[0].astype(float), 
                             transformed_channels[1].astype(float), 
                             transformed_channels[2].astype(float)])
    

    final_image += np.abs(np.min(final_image))
    final_image = final_image / np.max(final_image)

    return final_image

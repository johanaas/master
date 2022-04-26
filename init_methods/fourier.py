import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from utils.decision_function import decision_function
from utils.compute_distance import compute_distance
import cv2
import scipy.fftpack as fp
import copy

def create_fperturb_binary_seach_in_different_freq(img, model, params):
    
    # Create an adversarial image in frequency domain
    perturbed = create_fperurb_rgb(img, model, params)

    # Create masks for frequencies: Low, medium and High
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

    # Reduce L2 distance with binary search in frequency domain
    # Run binary search for frequnecies: Low, medium and High
    final_image1 = create_fperturb_binary_seach(low_mask, perturbed, img, model, params)
    final_image2 = create_fperturb_binary_seach(med_mask, final_image1, img, model, params)
    final_image3 = create_fperturb_binary_seach(high_mask, final_image2, img, model, params)

    return final_image3

def create_fperturb_binary_seach(mask, perturbed, img, model, params):
    # Upper and lower bound
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        # Compute binary search on each channel of RGB
        transformed_channels = []
        for i in range(3):
            # Fourier transform channel i of perturbed image
            rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, i])))

            # Retrive magnitude and phase from the transformation
            magnitude = np.abs(rgb_fft)
            phase = np.angle(rgb_fft, deg=False)
            
            # Fourier transform channel i of original image
            org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
            org_magnitude = np.abs(org_fft)

            # Binary search on the masks of the magnitudes of original and perturbed images
            blended = copy.deepcopy(magnitude)
            blended[mask] = (1 - mid) * org_magnitude[mask] + mid * magnitude[mask]

            # Inverse the Fourier Transformation with blended magnitude and phase
            b = blended*np.sin(phase)
            a = blended*np.cos(phase)
            z = a + b * 1j
            back_shift = np.fft.ifftshift(z)
            bb = np.fft.ifft2(back_shift).real

            transformed_channels.append(bb)

        # Retrive the blended image of the search
        final_image = np.dstack([transformed_channels[0].astype(float), 
                                transformed_channels[1].astype(float), 
                                transformed_channels[2].astype(float)])

        # Check if the search was successful
        # If the blended image is adversarial: True
        success = decision_function(model, final_image[None], params)

        if success:
            high = mid
        else:
            low = mid


    # We have found the optimal value of mid
    # mid = high to guanturee adversarial
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

    # Retrive the blended image with the value of high
    final_image = np.dstack([transformed_channels[0].astype(float), 
                                transformed_channels[1].astype(float), 
                                transformed_channels[2].astype(float)])

    return final_image

def create_fperurb_rgb(img, model, params):

    # Append noise in each channel of image
    transformed_channels = []
    for i in range(3):
        # Fourier transform channel i of image
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
        
        # Retrive magnitude and phase from the transformation
        magnitude = np.log(np.abs(rgb_fft))
        phase = np.angle(rgb_fft, deg=False)
        
        # Create masks based on the values of image to target all three frequencies
        high_mask = (magnitude < 1)
        med_mask = (magnitude > 2) | (magnitude < 4)
        low_mask = (magnitude > 5)

        # Append noise in each of the frequencies
        magnitude[high_mask] = np.random.uniform(0, 1, size = (224,224))[high_mask]
        magnitude[med_mask] = np.random.uniform(2, 4, size = (224,224))[med_mask]
        magnitude[low_mask] = np.random.uniform(5, 9, size = (224,224))[low_mask]

        # Inverse the Fourier Transformation with phase and perturbed magnitude 
        magnitude = np.exp(magnitude)
        b = magnitude*np.sin(phase)
        a = magnitude*np.cos(phase)
        z = a + b * 1j
        back_shift = np.fft.ifftshift(z)
        bb = np.fft.ifft2(back_shift).real

        transformed_channels.append(bb)
    
    # Retrive the blended image of the search
    final_image = np.dstack([transformed_channels[0].astype(float), 
                             transformed_channels[1].astype(float), 
                             transformed_channels[2].astype(float)])
    
    # Scale final_image since some values are < 0 and > 1
    final_image += np.abs(np.min(final_image))
    final_image = final_image / np.max(final_image)

    return final_image

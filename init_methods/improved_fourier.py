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
import jenkspy

def fourier_attack(img, model, params):
    
    perturbed, high_masks, mid_masks, low_masks = create_fperurb_rgb(img, model, params)

    """
    plt.imshow(perturbed)
    plt.title("Improved Perturbed image")
    plt.show()
    """

    """
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
    """

    # Reduce L2 distance with binary search in frequency domain
    # Run binary search for frequnecies: Low, medium and High

    final_image1 = create_fperturb_binary_seach(low_masks, perturbed, img, model, params)

    final_image2 = create_fperturb_binary_seach(mid_masks, final_image1, img, model, params)
    final_image3 = create_fperturb_binary_seach(high_masks, final_image2, img, model, params)

    """
    final_image3 += np.abs(np.min(final_image3))
    final_image3 = final_image3 / np.max(final_image3)

    plt.imshow(final_image3)
    plt.title("Improved After binary search")
    plt.show()
    """

    return final_image3

def create_fperturb_binary_seach(masks, perturbed, img, model, params):
    # Upper and lower bound
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        # Compute binary search on each channel of RGB
        transformed_channels = []
        for i in range(perturbed.shape[-1]):
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
            blended[masks[i]] = (1 - mid) * org_magnitude[masks[i]] + mid * magnitude[masks[i]]

            # Inverse the Fourier Transformation with blended magnitude and phase
            b = blended*np.sin(phase)
            a = blended*np.cos(phase)
            z = a + b * 1j
            back_shift = np.fft.ifftshift(z)
            bb = np.fft.ifft2(back_shift).real

            """
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(np.log(org_magnitude))
            axs[0].set_title("org")
            axs[1].imshow(np.log(blended))
            axs[1].set_title("bs")
            plt.show()
            """

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
    for i in range(perturbed.shape[-1]):
        rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, i])))
        magnitude = np.abs(rgb_fft)

        org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
        org_magnitude = np.abs(org_fft)
        blended = copy.deepcopy(magnitude)
        blended[masks[i]] = (1 - high) * org_magnitude[masks[i]] + high * magnitude[masks[i]]


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

    transformed_channels = []
    high_masks = []
    mid_masks = []
    low_masks = []
    for i in range(img.shape[-1]):
        channel = img[:, :, i]
        rgb_fft = np.fft.fftshift(np.fft.fft2((channel)))
        
        # Retrive magnitude and phase from the transformation
        magnitude = np.log(np.abs(rgb_fft))
        phase = np.angle(rgb_fft, deg=False)
        
        # Create masks based on the values of image to target all three frequencies
        high_mask = (magnitude < 1)
        med_mask = (magnitude > 2) * (magnitude < 4)
        low_mask = (magnitude > 5)

        # Append noise in each of the frequencies
        magnitude[high_mask] = np.random.uniform(0, 1, channel.shape)[high_mask]
        
        magnitude[med_mask] = np.random.uniform(2, 4, channel.shape)[med_mask]
        
        magnitude[low_mask] = np.random.uniform(5, 9, channel.shape)[low_mask]

        # Inverse the Fourier Transformation with phase and perturbed magnitude 
        magnitude = np.exp(magnitude)
        b = magnitude*np.sin(phase)
        a = magnitude*np.cos(phase)
        z = a + b * 1j
        
        bb = np.fft.ifft2(np.fft.ifftshift(z)).real

        transformed_channels.append(bb)
        low_masks.append(low_mask)
        mid_masks.append(med_mask)
        high_masks.append(high_mask)

        """
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(low_mask)
        axs[0].set_title("Low")
        axs[1].imshow(med_mask)
        axs[1].set_title("Mid")
        axs[2].imshow(high_mask)
        axs[2].set_title("High")
        plt.show()
        """
    
    # Retrive the blended image of the search
    final_image = np.dstack([transformed_channels[0].astype(float), 
                             transformed_channels[1].astype(float), 
                             transformed_channels[2].astype(float)])
    
    # Scale final_image since some values are < 0 and > 1
    #final_image += np.abs(np.min(final_image))
    #final_image = final_image / np.max(final_image)

    return final_image, high_masks, mid_masks, low_masks

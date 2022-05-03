import numpy as np
from utils.decision_function import decision_function
from utils.compute_distance import compute_distance
import cv2
import copy
from init_methods.par import get_par_patches
from utils.binary_search import binary_search

def FCBSA(model, sample, params):

    freq_bands = ["low", "medium", "high"]

    # Identify radius r_l and r_h
    # TODO: Find r_l and r_h dynamically
    r_l = 25
    r_h = 75

    # Create mask for each frequency band
    mask = np.zeros((224,224))
    cv2.circle(mask, (112,112), r_l, 1, -1)
    params["low"] = (mask == 1)

    mask = np.zeros((224,224))
    cv2.circle(mask, (112,112), r_h, 1, -1)
    cv2.circle(mask, (112,112), r_l, 0, -1)
    params["medium"] = (mask == 1)

    mask = np.zeros((224,224))
    cv2.circle(mask, (112,112), r_h, 1, -1)
    params["high"] = (mask == 0)

    # Create initial perturbation
    success = 0
    num_evals = 0

    if params['target_label'] is None:
        # Find a misclassified random noise.
        while True:
            perturbed, success = initial_perturbation(model, sample, params)
            num_evals += 1
            if success:
                break
            assert num_evals < 1e4,"Initialization failed! "
            "Use a misclassified image as `target_image`" 
    
    # Binary search in the different frequency bands
    for band in freq_bands:
        perturbed = frequnecy_band_binary_search(params[band], perturbed, sample, model, params)
    
    # Perform PAR
    perturbed = get_par_patches(sample, model, params, noise=np.copy(perturbed), plot_each_step=False)

    # Last BS
    perturbed = binary_search(model, sample, perturbed, params)

    print("L2: ", np.linalg.norm(sample - perturbed))

    return perturbed

def frequnecy_band_binary_search(band, perturbed, img, model, params):
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
            blended[band] = (1 - mid) * org_magnitude[band] + mid * magnitude[band]

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
        success = decision_function(model, final_image[None], params, img)

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
            blended[band] = (1 - high) * org_magnitude[band] + high * magnitude[band]


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


def initial_perturbation(model, img, params):

    # Append noise in each channel of RGB image
    transformed_channels = []
    for i in range(3):
        # Fourier transform channel i of image
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
        
        # Retrive magnitude and phase from the transformation
        magnitude = np.log(np.abs(rgb_fft))
        phase = np.angle(rgb_fft, deg=False)

        # Append noise in each of the frequencies
        magnitude[params["high"]] = np.random.uniform(0, 1, size = (224,224))[params["high"]]
        magnitude[params["medium"]] = np.random.uniform(2, 4, size = (224,224))[params["medium"]]
        magnitude[params["low"]] = np.random.uniform(5, 9, size = (224,224))[params["low"]]

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
    
    success = decision_function(model, final_image[None], params, img)

    return final_image, success

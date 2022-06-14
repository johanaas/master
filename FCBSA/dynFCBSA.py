import numpy as np
from utils.decision_function import decision_function
from utils.compute_distance import compute_distance
import cv2
import copy
from init_methods.par import get_par_patches
from utils.binary_search import binary_search
from utils.truncnorm import get_truncated_normal
import config as CFG

from matplotlib import pyplot as plt

from utils.clip_image import clip_image

def dynFCBSA(model, sample, params, alpha = 0.5):

    freq_bands = ["low", "medium", "high"]
    print("Start: ", np.argmax(model.predict(sample)))
    original_label = np.argmax(model.predict(sample))

    # Identify radius r_l and r_h
    freq_params = select_freq_params(sample, alpha)

    # Create mask for each frequency band
    masks = create_masks(freq_params)

    # Create initial perturbation
    success = 0
    num_evals = 0

    if params['target_label'] is None:
        # Find a misclassified random noise.
        perturb_phase = False
        while True:
            perturbed, success = initial_perturbation(model, sample, params, freq_params, masks, perturb_phase)
            num_evals += 1
            if success:
                if perturb_phase:
                    print("\n\nPhase was also perturbed!\n\n")
                break
            perturb_phase = True
            assert num_evals < 1e4,"Initialization failed! "
            "Use a misclassified image as `target_image`" 


    # Binary search in the different frequency bands

    perturbed = frequnecy_band_binary_search("low", perturbed, sample, model, params, freq_params, masks)
    
    perturbed = frequnecy_band_binary_search("medium", perturbed, sample, model, params, freq_params, masks)
    
    perturbed = frequnecy_band_binary_search("high", perturbed, sample, model, params, freq_params, masks)

    # Perform PAR
    perturbed = get_par_patches(sample, model, params, noise=np.copy(perturbed), plot_each_step=False)
    perturbed = clip_image(perturbed, params['clip_min'], params['clip_max'])

    # Last BS
    perturbed = binary_search(model, sample, perturbed, params)
    perturbed = clip_image(perturbed, params['clip_min'], params['clip_max'])

    return perturbed

def frequnecy_band_binary_search(band, perturbed, img, model, params, freq_params, masks):
    
   
    # Upper and lower bound
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0

        # Compute binary search on each channel of RGB
        transformed_channels = []
        for k, v in freq_params.items():
            # Fourier transform channel i of perturbed image
            rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, k])))

            # Retrive magnitude and phase from the transformation
            magnitude = np.abs(rgb_fft)
            phase = np.angle(rgb_fft, deg=False)
            
            # Fourier transform channel i of original image
            org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, k])))
            org_magnitude = np.abs(org_fft)

            # Binary search on the masks of the magnitudes of original and perturbed images
            blended = copy.deepcopy(magnitude)
            blended[masks[k][band]] = (1 - mid) * org_magnitude[masks[k][band]] + mid * magnitude[masks[k][band]]

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
        final_image = clip_image(final_image, params['clip_min'], params['clip_max'])
        success = decision_function(model, final_image[None], params, img)
        
        if success:
            high = mid
        else:
            low = mid


    # We have found the optimal value of mid
    # mid = high to guanturee adversarial
    transformed_channels = []
    for k, v in freq_params.items():
            rgb_fft = np.fft.fftshift(np.fft.fft2((perturbed[:, :, k])))
            magnitude = np.abs(rgb_fft)

            org_fft = np.fft.fftshift(np.fft.fft2((img[:, :, k])))
            org_magnitude = np.abs(org_fft)
            blended = copy.deepcopy(magnitude)
            blended[masks[k][band]] = (1 - high) * org_magnitude[masks[k][band]] + high * magnitude[masks[k][band]]

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

    final_image = clip_image(final_image, params['clip_min'], params['clip_max'])

    if high == 1.0:
        print("setting old value")
        print("****************************************************")
        final_image = perturbed
    success = decision_function(model, final_image[None], params, img)
    assert success == 1
    
    return final_image


def initial_perturbation(model, img, params, freq_params, masks, perturb_phase=False):

    # Append noise in each channel of RGB image
    transformed_channels = []
    for k, v in freq_params.items():
        # Fourier transform channel i of image
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, k])))
        
        # Retrive magnitude and phase from the transformation
        magnitude = np.log(np.abs(rgb_fft))
        phase = np.angle(rgb_fft, deg=False)
        
        min_magnitude = np.min(magnitude)
        max_magnitude = np.max(magnitude)
        channel_mean = freq_params[k]["mean"]
        channel_std = freq_params[k]["sigma"]
        
        # Append noise in each of the frequencies
        high_distribution = get_truncated_normal(mean=channel_mean, sd=channel_std, low=min_magnitude, high=freq_params[k]["left_clip_tail"], shape=magnitude.shape, mask=masks[k]["high"])
        magnitude[masks[k]["high"]] = high_distribution[masks[k]["high"]]

        medium_distribution = get_truncated_normal(mean=channel_mean, sd=channel_std, low=freq_params[k]["left_clip_tail"], high=freq_params[k]["right_clip_tail"], shape=magnitude.shape, mask=masks[k]["medium"])
        magnitude[masks[k]["medium"]] = medium_distribution[masks[k]["medium"]]  

        low_distribution = get_truncated_normal(mean=channel_mean, sd=channel_std, low=freq_params[k]["right_clip_tail"], high=max_magnitude, shape=magnitude.shape, mask=masks[k]["low"])
        magnitude[masks[k]["low"]] = low_distribution[masks[k]["low"]]  

        if perturb_phase:
            phase = np.random.uniform(0, np.pi, phase.shape)

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
    final_image += np.abs(np.min(final_image))
    final_image = final_image / np.max(final_image)

    success = decision_function(model, final_image[None], params, img)

    return final_image, success

def create_masks(freq_params):

    masks = {
        0: {
            "low": None,
            "medium": None,
            "high": None,
        },
        1: {
            "low": None,
            "medium": None,
            "high": None,
        },
        2: {
            "low": None,
            "medium": None,
            "high": None,
        }
    }

    for k, v in masks.items():
        mask = np.zeros((224,224))
        cv2.circle(mask, (112,112), freq_params[k]["radius"][1], 1, -1)
        v["low"] = (mask == 1)

        mask = np.zeros((224,224))
        cv2.circle(mask, (112,112), freq_params[k]["radius"][0], 1, -1)
        cv2.circle(mask, (112,112), freq_params[k]["radius"][1], 0, -1)
        v["medium"] = (mask == 1)

        mask = np.zeros((224,224))
        cv2.circle(mask, (112,112), freq_params[k]["radius"][0], 1, -1)
        v["high"] = (mask == 0)
    
    return masks


def select_freq_params(img, alpha):
    freq_params = {
        0: {
            "radius": (),
            "right_clip_tail": -1,
            "left_clip_tail": -1,
            "mean": -1,
            "sigma": -1
        },
        1: {
            "radius": (),
            "right_clip_tail": -1,
            "left_clip_tail": -1,
            "mean": -1,
            "sigma": -1
        },
        2: {
            "radius": (),
            "right_clip_tail": -1,
            "left_clip_tail": -1,
            "mean": -1,
            "sigma": -1
        },
    }

    for k, v in freq_params.items():
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, k])))
        magnitude = np.log(np.abs(rgb_fft))
        explore_mag = magnitude.flatten()
        standard_dev = np.std(explore_mag)
        origin = np.average(explore_mag)
        v["mean"] = origin
        v["sigma"] = standard_dev
        v["right_clip_tail"] = origin + standard_dev * 3
        v["left_clip_tail"] = origin - standard_dev * 2
        

        right_freq_tail = (magnitude > v["right_clip_tail"]).astype(int)
        r_low = int(np.average(find_dist_to_center(right_freq_tail)))

        left_freq_tail = (magnitude < v["left_clip_tail"]).astype(int)
        r_high = int(np.average(find_dist_to_center(left_freq_tail)))

        v["radius"] = (r_high, r_low)


    return freq_params


def find_dist_to_center(testdwa):
    center_dist = []

    for k in range(len(testdwa)):
        for l in range(len(testdwa)):
            if testdwa[k][l] == 1:
                center_dist.append(abs(np.sqrt( (112 - k)**2 + (112 - l)**2 )))
    return center_dist
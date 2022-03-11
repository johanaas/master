import numpy as np
from utils import decision_function

def deconvolute(original_image, noisy_image, model, params):

    mask = (original_image[:, :, :] != noisy_image[:, :, :])

    for blend in range(0.1, 1.1, 0.1):
        img = np.copy(original_image)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.any(mask[i, j]):
                    spread(img, i, j, noisy_image[i, j, :], blend=blend)
        
        if decision_function(model, img[None], params)[0]:
            return img
            
    print("Deconvolute could not find an adversarial image. Using the same image as PAR.")
    return original_image



def spread(img, x, y, value, blend=0.5, filter_size=3):
    min_x = 0
    min_y = 0
    max_x = img.shape[0]
    max_y = img.shape[1]

    for i in range(filter_size):
        for j in range(filter_size):
            filter_pixel_x = x - 1 + i
            filter_pixel_y = y - 1 + j

            if not min_x <= filter_pixel_x < max_x or not min_y <= filter_pixel_y < max_y:
                continue

            img[filter_pixel_x, filter_pixel_y, :] = (1 - blend) * img[filter_pixel_x, filter_pixel_y, :] + blend * value

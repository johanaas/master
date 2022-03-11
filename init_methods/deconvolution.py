import numpy as np

def deconvolute(original_image, noisy_image):
    img = np.copy(original_image)

    mask = (img[:, :, :] != noisy_image[:, :, :])

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if np.any(mask[i, j]):
                spread(img, i, j, noisy_image[i, j, :])
    return img

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

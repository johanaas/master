from matplotlib import pyplot as plt
import cv2
import numpy as np
from utils.decision_function import decision_function
import query_counter
import evaluation
from utils.compute_distance import compute_distance

def get_par_patches(img, model, params, noise=[], plot_each_step=False):
    """Returns random noise on important patches identified
    through querying the model. The returned image is between
    [0, 1] and is the noise overlayed on the original image

    Args:
        img (nd.array): Original image
        model (_type_): Model to query
        params (_type_): HSJA params
        plot_each_step (bool, optional): Whether to plot each stage of PAR or not. Defaults to False.

    Returns:
        nd.array: An adversarial version of the original image with noise patches
    """
    assert img.shape[0] == img.shape[1]
    img_width = img.shape[0]

    if len(noise) != 0:
        noise = noise
    else:
        noise = np.random.uniform(0, 1, (img.shape))

    saved_patches = [(0, 0, img_width)]

    i = 0

    new_noise = noise
    while len(saved_patches) > 0:
        saved_patch = saved_patches.pop(0)
        new_patches, new_noise = remove_noise(img, new_noise, saved_patch, model, params, plot_each_step=plot_each_step)
        if new_patches != None:
            for p in new_patches:
                saved_patches.append(p)
        i += 1

    return new_noise


def remove_noise(img, noise, patch, model, params, plot_each_step=False):
    start_i, start_j, region_size = patch

    if region_size % 2 != 0:
        return None, noise

    patch_size = region_size / 2
    num_patches = int(region_size // patch_size)

    base_x = start_i
    base_y = start_j

    out_patches = []

    for i in range(num_patches):
        for j in range(num_patches):
            start_x = int(base_x + i * patch_size)
            end_x = int(base_x + (i+1) * patch_size)
            start_y = int(base_y + j * patch_size)
            end_y = int(base_y + (j+1) * patch_size)

            # Remove noise in the area to check
            noise_img = np.copy(noise)
            noise_img[start_x:end_x, start_y:end_y, :] = img[start_x:end_x, start_y:end_y, :]

            # Predict with noise removed on the area to check
            if decision_function(model,noise_img[None], params, img)[0]: #np.argmax(model.predict(noise_img)) != params["original_label"]:
                # Original image in this patch is adversarial
                # We should remove the noise in this patch
                noise[start_x:end_x, start_y:end_y, :] = img[start_x:end_x, start_y:end_y, :]
                if plot_each_step:
                    visualize_noise_img = np.copy(noise_img)
                    visualize_noise_img[start_x:end_x, start_y:end_y, :] = np.asarray([0, 0, 0])
                    plt.imshow(visualize_noise_img)
                    plt.title("{}".format(("Remove noise", (start_x, start_y), (end_x, end_y))))
                    plt.show()
            else:
                # Original image in this patch is not adversarial
                # We should keep the noise in this patch and save
                # it for further searching
                out_patches.append((start_x, start_y, patch_size))
                if plot_each_step:
                    visualize_noise_img = np.copy(noise_img)
                    visualize_noise_img[start_x:end_x, start_y:end_y, :] = np.asarray([0, 0, 0])
                    plt.imshow(visualize_noise_img)
                    plt.title("{}".format(("Keep noise", (start_x, start_y), (end_x, end_y))))
                    plt.show()

    if len(out_patches) > 0:
        return out_patches, noise
    return None, noise

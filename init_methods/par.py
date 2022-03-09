from matplotlib import pyplot as plt
import cv2
import numpy as np
from init_methods.utils import decision_function
import settings

#############################################################################
#                                                                           #
#                                                                           #
#                                   PAR                                     #
#                                                                           #
#                                                                           #
#############################################################################

def get_par_patches(img, model, params, plot_each_step=False):
    """ Returns random noise on important patches identified
    through querying the model. The returned image is between
    [0, 1] and is the noise overlayed on the original image
    """
    assert img.shape[0] == img.shape[1]
    img_width = img.shape[0]
    np.random.seed(69)
    random_noise = np.random.uniform(0, 1, (img.shape))

    saved_patches = [(0, 0, img_width)]

    i = 0

    new_noise = random_noise
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

            if settings.queries >= settings.circle_queries:
                break

            # Predict with noise removed on the area to check
            if decision_function(model,noise_img[None], params)[0]: #np.argmax(model.predict(noise_img)) != params["original_label"]:
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


#############################################################################
#                                                                           #
#                                                                           #
#                               Saliency map                                #
#                                                                           #
#                                                                           #
#############################################################################


def compute_saliency_filter(
    sample,
    threshold=0.1,
    clip_max=1
):
    """ Computes the saliency map for the input image
    Input image is 3-dimensional with channels last and between [0, 255]


    Returns:
        - Threshold map 
    """

    sample = np.float32(sample)

    sal_map, thresh_map = get_saliency_map(sample, threshold=threshold, max_value=clip_max)

    idx = thresh_map[:, :] != 0

    sample[idx] = np.asarray([np.random.uniform(0, clip_max, 3) for _ in sample[idx]])
    
    return sample

    

def get_saliency_map(img, threshold=0.5, max_value=1):
    """Calculates the salicency map and threshold map for a given image.

    Channels last is required in the image.

    Args:
        img: numpy.ndarray
    """
    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(img)
    saliency_map = (saliency_map).astype("float32")
    _, thresh_map = cv2.threshold(saliency_map, threshold, max_value, cv2.THRESH_BINARY)

    return saliency_map, thresh_map


#############################################################################
#                                                                           #
#                                                                           #
#                               Random utils                                #
#                                                                           #
#                                                                           #
#############################################################################

def plot_image(img, title=None):
    plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.show()

def plot_checking_area(img, start, end):
    img_copy = np.copy(img)
    img_copy[start[0]:end[0], start[1]:end[1], :] = np.asarray([0, 0, 0])
    plot_image(img_copy, title="Area to check")

def load_image(path, max_value=1):
    """Read image. Returns RGB image between [0, max_value]
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img / 255) * max_value

def preprocess_image(img, shape=(224, 224)):
    """Center crop the image to shape
    """
    h, w, c = img.shape
    new_w, new_h = shape

    start_w = int(w // 2 - int(new_w // 2))
    start_h = int(h // 2 - int(new_h // 2))

    end_w = start_w + new_w
    end_h = start_h + new_h

    return img[start_h:end_h, start_w:end_w]

class Model():
    def __init__(self, percent_adv):
        self.percent_adv = percent_adv

    def predict(self, img):
        return np.random.uniform() < self.percent_adv
from build_model import ResnetModel50
from hsja import hsja
import numpy as np
from load_imagenet import load_imagenet
from matplotlib import pyplot as plt
import settings
from datetime import datetime
from init_methods.random_init import get_random_noise
from init_methods.par import get_par_patches
from init_methods.fourier import fourier_transform_rgb
from init_methods.utils import binary_search, Logger, compute_distance, decision_function
from imagenet_classes import class_names
import sys
import cv2


def show_fourier(img, perturbed):
    f = np.fft.fft2(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY))
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(f_shift))
    plt.imshow(magnitude_spectrum)
    plt.show()

    f = np.fft.fft2(cv2.cvtColor(np.float32(perturbed), cv2.COLOR_RGB2GRAY))
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum2 = 20*np.log(np.abs(f_shift))
    plt.imshow(magnitude_spectrum2)
    plt.show()

    f_diff = np.abs(magnitude_spectrum2) - np.abs(magnitude_spectrum)
    plt.imshow(f_diff)
    plt.show()

    f_mask = (f_diff > 10)
    magnitude_spectrum[f_mask] = magnitude_spectrum2[f_mask]

    #Convert back



def run_hsja(model, samples, sample_perturbed):


    target_label = None
    target_image = None

    #print('attacking the {}th sample...'.format(i))

    perturbed = hsja(model, 
                        sample,
                        sample_perturbed,
                        clip_max = 1, 
                        clip_min = 0, 
                        constraint = "l2", 
                        num_iterations = 25, 
                        gamma = 1.0, 
                        target_label = None, 
                        target_image = None, 
                        stepsize_search = "geometric_progression", 
                        max_num_evals = 1e4,
                        init_num_evals = 100)
    return perturbed

def run_init_method(experiment, sample, model, params):

    if experiment == "random":
        start_image = get_random_noise(model, params)
    elif experiment == "naive":
        start_image = get_naive_noise(sample, model, params)
    elif experiment == "fourier":
        start_image = fourier_transform_rgb(sample, model, params)
    elif experiment == "par":
        start_image = get_par_patches(sample, model, params)
        #pred_label = np.argmax(model.predict(start_image))
    
    return start_image



if __name__ == '__main__':
    
    # Printing to logfile
    sys.stdout = Logger(sys.stdout, 'logfile.txt')

    # Start counter of queries
    settings.init_queries()
    settings.max_queries = 250


    model = ResnetModel50() # [ ResnetModel50(), ResnetModel101() ]
    x_test = load_imagenet(4)

    experiments = ["fourier", "random", "par"]

    eval_start = {}
    eval_end = {}

    for experiment in experiments:
        eval_start[experiment] = []
        eval_end[experiment] = []

    

    fig, axs = plt.subplots(len(experiments))
    
    for i, sample in enumerate(x_test):
        original_label = np.argmax(model.predict(sample))
        print("-----------------------------------------------")
        print("Attacking sample nr {} / {}".format(i+1, len(x_test)))
        print("Time: ", datetime.now())

        params = {
                "original_label": original_label,
                "target_label": None,
                "clip_min": 0,
                "clip_max": 1,
                "shape": (224,224,3)
        }

        for j, experiment in enumerate(experiments):
            # Resetting query counter for each experiment
            settings.queries = 0
            
            print("\nExperiment {} \n".format(experiment))

            # Running init_method based on experiment
            start_image = run_init_method(experiment, sample, model, params)
            print("Adversarial before BS: ", decision_function(model,start_image[None], params)[0])
            assert decision_function(model,start_image[None], params)[0] == True

            # Look at fourier:
            #show_fourier(sample, start_image)



            # Conduct Binary Search
            bs_img = binary_search(model, sample, start_image, params)
            init_dist = compute_distance(bs_img, sample)
            eval_start[experiment].append(init_dist)
            print("Init ditance: ", init_dist)

            # Run HSJA attack method
            final_img = bs_img#run_hsja(model, sample, bs_img)
            eval_end[experiment].append(compute_distance(final_img, sample))
            
            # Saving computed images to folder /results
            result_image = np.concatenate([sample, np.zeros((sample.shape[0],8,3)), start_image, np.zeros((sample.shape[0],8,3)), bs_img, np.zeros((sample.shape[0],8,3)), final_img], axis = 1)
            axs[j].imshow(result_image)
            #plt.imshow(result_image)
            axs[j].title.set_text("Original image ({}) - After Init by {} ({}) - After Binary Search ({}) - After HSJA ({})".format(
                class_names[original_label],
                experiment, 
                class_names[np.argmax(model.predict(start_image))], 
                class_names[np.argmax(model.predict(bs_img))], 
                class_names[np.argmax(model.predict(final_img))]))
        plt.show()

        for exp in experiments:
            print("Image number {} / {}".format(i+1, len(x_test)))
            print("Start mean for experiment {}: ".format(exp), np.median(eval_start[exp]))
            print("End mean for experiment {}: ".format(exp), np.median(eval_end[exp]))
            print("Start avg for experiment {}: ".format(exp), np.mean(eval_start[exp]))
            print("End avg for experiment {}: ".format(exp), np.mean(eval_end[exp]))
            print("-------------------")

        #fig.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
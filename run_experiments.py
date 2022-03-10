import config as CFG

import numpy as np
import random
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

from init_methods import get_start_image
from models import get_model

from HSJA.hsja import hsja
from load_imagenet import load_imagenet
from matplotlib import pyplot as plt
import settings
from datetime import datetime
from utils import binary_search, Logger, compute_distance
from imagenet_classes import class_names
import sys

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

if __name__ == '__main__':
    
    # Printing to logfile
    sys.stdout = Logger(sys.stdout, 'logs/logfile.txt')

    # Start counter of queries
    settings.init_queries()

    model = get_model(CFG.MODEL)

    x_test = load_imagenet(num_images=CFG.NUM_IMAGES)

    experiments = CFG.EXPERIMENTS

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
            start_image = get_start_image(
                experiment=experiment,
                sample=sample,
                model=model,
                params=params)
            

            # Conduct Binary Search
            bs_img = binary_search(model, sample, start_image, params)
            init_dist = compute_distance(bs_img, sample)
            eval_start[experiment].append(init_dist)
            print("Init ditance: ", init_dist)

            # Run HSJA attack method
            final_img = run_hsja(model, sample, bs_img)
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
            #plt.show()

        for exp in experiments:
            print("Image number {} / {}".format(i+1, len(x_test)))
            print("Start mean for experiment {}: ".format(exp), np.median(eval_start[exp]))
            print("End mean for experiment {}: ".format(exp), np.median(eval_end[exp]))
            print("Start avg for experiment {}: ".format(exp), np.mean(eval_start[exp]))
            print("End avg for experiment {}: ".format(exp), np.mean(eval_end[exp]))
            print("-------------------")

        fig.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
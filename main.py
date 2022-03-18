import config as CFG

import numpy as np
import random
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

from init_methods import get_start_image
from models import get_model
from datasets import get_dataset
from evaluation import start_eval_experiment, init_plotting, add_dist_queries, plot_all_experiments, plot_median, save_experiment_image

from HSJA.hsja import hsja
from matplotlib import pyplot as plt
import query_counter
from datetime import datetime
from utils import binary_search, Logger, compute_distance
from imagenet_classes import class_names
import sys
import os

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
    
    # Initiation from config
    model = get_model(CFG.MODEL)
    data = get_dataset(CFG.DATASET, CFG.NUM_IMAGES)
    experiments = CFG.EXPERIMENTS

    if CFG.RUN_EVAL:
        exp_dir = "results/{}_{}_cap{}_imgs{}_{}_{}".format(
            datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
            "_".join(CFG.EXPERIMENTS),
            CFG.MAX_NUM_QUERIES,
            CFG.NUM_IMAGES,
            CFG.DATASET,
            CFG.MODEL
            )
        # Make new dir with experiement name
        os.mkdir(exp_dir)
        os.mkdir(exp_dir + "/attack_process_imgs")
        # Write every print statement to logfile for experiement
        sys.stdout = Logger(sys.stdout, exp_dir + "/logfile.txt")
        # Init data collection
        query_counter.init_queries()
        init_plotting(experiments)


    eval_start = {}
    eval_end = {}

    for experiment in experiments:
        eval_start[experiment] = []
        eval_end[experiment] = []

    

    #fig, axs = plt.subplots(len(experiments))
    
    for i, sample in enumerate(data):
        original_label = np.argmax(model.predict(sample))
        print("-----------------------------------------------")
        print("Attacking sample nr {} / {}".format(i+1, CFG.NUM_IMAGES))
        print("Time: ", datetime.now())

        params = {
                "original_label": original_label,
                "target_label": None,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "shape": sample.shape
        }

        for j, experiment in enumerate(experiments):
            # Resetting query counter for each experiment
            query_counter.queries = 0
            start_eval_experiment(experiment)

            
            print("\nExperiment {} \n".format(experiment))

            # Running init_method based on experiment
            init_pert = get_start_image(
                experiment=experiment,
                sample=sample,
                model=model,
                params=params)
            
            # Conduct Binary Search
            boundary_pert = binary_search(model, sample, init_pert, params)
            init_dist = compute_distance(boundary_pert, sample)
            eval_start[experiment].append(init_dist)
            print("Init ditance: ", init_dist)
            add_dist_queries(init_dist)

            # Run HSJA attack method
            final_pert = run_hsja(model, sample, boundary_pert)
            eval_end[experiment].append(compute_distance(final_pert, sample))
            
            save_experiment_image(model, exp_dir, i+1, sample, init_pert, boundary_pert, final_pert)

        for exp in experiments:
            print("Image number {} / {}".format(i+1, CFG.NUM_IMAGES))
            print("Start mean for experiment {}: ".format(exp), np.median(eval_start[exp]))
            print("End mean for experiment {}: ".format(exp), np.median(eval_end[exp]))
            print("Start avg for experiment {}: ".format(exp), np.mean(eval_start[exp]))
            print("End avg for experiment {}: ".format(exp), np.mean(eval_end[exp]))
            print("-------------------")
        
    plot_all_experiments(exp_dir, experiments)
    plot_median(exp_dir, experiments)
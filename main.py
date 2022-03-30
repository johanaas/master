import config as CFG

import numpy as np
import random
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

from init_methods import get_start_image
from models import get_model
from datasets import get_dataset
from evaluation import start_eval_experiment, init_plotting, add_dist_queries, plot_all_experiments, plot_median

from HSJA.hsja import hsja
from matplotlib import pyplot as plt
import query_counter
from datetime import datetime
from utils import binary_search, Logger, compute_distance, decision_function
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
    
    if CFG.LOG_DIR is not None:
      logfile_path = "{}/{}_cap{}_{}imgs_{}_{}.txt".format(
        CFG.LOG_DIR,
        "_".join(CFG.EXPERIMENTS),
        CFG.MAX_NUM_QUERIES,
        CFG.NUM_IMAGES,
        CFG.DATASET,
        CFG.MODEL
      )
      sys.stdout = Logger(sys.stdout, logfile_path)

    # Start counter of queries
    query_counter.init_queries()

    model = get_model(CFG.MODEL)

    data = get_dataset(CFG.DATASET, CFG.NUM_IMAGES)

    experiments = CFG.EXPERIMENTS

    init_plotting(experiments)

    eval_start = {}
    eval_end = {}

    for experiment in experiments:
        eval_start[experiment] = []
        eval_end[experiment] = []

    best_exp = [0]*len(experiments)
    non_adv_counter = 0
    throwaway_counter = 0
    fpar_sum_queries = 0

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

        random_boundary_dist = 0

        for j, experiment in enumerate(experiments):
            # Resetting query counter for each experiment
            query_counter.queries = 0
            start_eval_experiment(experiment)

            
            #print("\nExperiment {} \n".format(experiment))

            # Running init_method based on experiment
            start_image = get_start_image(
                experiment=experiment,
                sample=sample,
                model=model,
                params=params)

            #if start_image.shape[0] == 0:
            #    print("\n\n\nImage {} from {} not adversarial!\n\n\n".format(j, experiment))
            #    eval_start[experiment].append(np.Inf)
            #    eval_end[experiment].append(np.Inf)
            #    non_adv_counter += 1
            #    continue
            if not decision_function(model,start_image[None], params)[0]:
                print("\n\n\nImage {} from {} not adversarial!\n\n\n".format(j, experiment))
                continue

            init_dist = compute_distance(start_image, sample)

            eval_start[experiment].append(init_dist)

            
            # Conduct Binary Search
            boundary_img = binary_search(model, sample, start_image, params)
            boundary_dist = compute_distance(boundary_img, sample)

            bs_dist = compute_distance(boundary_img, sample)
            eval_end[experiment].append(bs_dist)

            if experiment == "fpar":
                query_counter.max_queries = query_counter.queries
                fpar_sum_queries += query_counter.queries

            print("Queries used: {} -> {}".format(experiment, query_counter.queries))

            """
            if experiment == "random":
                random_boundary_dist = boundary_dist
            else:
                #print("Random at {} vs Fourier at {}".format(random_boundary_dist, boundary_dist))
                if boundary_dist > random_boundary_dist:
                    # Fourier have higher L2
                    # We dont do HSJA
                    #print("\n\n\nFourier have higher L2: Skipping HSJA\n\n\n")

                    continue
            #eval_end[experiment].append(boundary_dist)
            #print("Init ditance: ", init_dist)
            #add_dist_queries(init_dist)
            """

            if experiment == "random":
                # Run HSJA attack method
                final_img = run_hsja(model, sample, boundary_img)
                eval_end[experiment].append(compute_distance(final_img, sample))
            

            # Saving computed images to folder /results
            #result_image = np.concatenate([sample, np.zeros((sample.shape[0],8,3)), start_image, np.zeros((sample.shape[0],8,3)), bs_img, np.zeros((sample.shape[0],8,3)), final_img], axis = 1)
            #axs[j].imshow(result_image)
            #plt.imshow(result_image)
            #axs[j].title.set_text("Original image ({}) - After Init by {} ({}) - After Binary Search ({}) - After HSJA ({})".format(
            #    class_names[original_label],
            #    experiment, 
            #    class_names[np.argmax(model.predict(start_image))], 
            #    class_names[np.argmax(model.predict(bs_img))], 
            #    class_names[np.argmax(model.predict(final_img))]))
            #plt.show()

        
        """
        for exp in experiments:
            print("Image number {} / {}".format(i+1, CFG.NUM_IMAGES))
            print("Start mean for experiment {}: ".format(exp), np.median(eval_start[exp]))
            print("End mean for experiment {}: ".format(exp), np.median(eval_end[exp]))
            print("Start avg for experiment {}: ".format(exp), np.mean(eval_start[exp]))
            print("End avg for experiment {}: ".format(exp), np.mean(eval_end[exp]))
            print("-------------------")
        """
        
        start_distances = []
        end_distances = []
        for k in eval_start:
            start_distances.append(eval_start[k][-1])
            end_distances.append(eval_end[k][-1])

        print("Start distances:", start_distances)
        print("End distances:", end_distances)

        """
        if start_distances[0] == np.max(start_distances) start_distances[1]:
            print("Random started better than PAR, throwing away...")
            throwaway_counter += 1
            continue
        """

        lowest = np.argmin(end_distances)
        print("Best method:", experiments[lowest])
        best_exp[lowest] += 1
        print("Experiments:", experiments)
        print("{}/{}:".format(str(i+1).rjust(len(str(CFG.NUM_IMAGES)), "0"), CFG.NUM_IMAGES), best_exp)
        print("Number of non adversarial images:", non_adv_counter)
        print("Fpar mean queries:", fpar_sum_queries / (i+1))
        #print("Throwaways:", throwaway_counter)
        #for l, exp_num in enumerate(min_dists):
        #    print("{}:\t{}".format(str(l).rjust(4, "0"), exp_num))
        
    #plot_all_experiments(experiments)
    #plot_median(experiments)

    print("Experiments:", experiments)
    print("Best:", best_exp)
    

        #fig.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
import config as CFG

import numpy as np
import random
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

import query_counter
from init_methods import get_start_image
from models import get_model
from datasets import get_dataset
from evaluation import start_eval_experiment, init_plotting, add_dist_queries, plot_all_experiments, plot_median

from utils.run_hsja import run_hsja
from utils.binary_search import binary_search
from utils.compute_distance import compute_distance
from utils.logging import setup_logging
from utils.printing import print_current_medians_and_averages, print_sample_progress, print_iteration_summary

from matplotlib import pyplot as plt
from utils.decision_function import decision_function

import sys

if __name__ == '__main__':
    
    if CFG.LOG_DIR is not None:
        setup_logging()

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

    fpar_max_queries = 0
    fpar_total_queries = 0
    random_total_queries = 0

    for i, sample in enumerate(data):

        print_sample_progress(
            current_iteration=i+1,
            max_iteration=CFG.NUM_IMAGES,
            show_time=True)

        original_label = np.argmax(model.predict(sample))

        params = {
                "original_label": original_label,
                "target_label": None,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "shape": sample.shape
        }


        for j, experiment in enumerate(experiments):

            query_counter.reset_queries()
            start_eval_experiment(experiment)


            start_image = get_start_image(
                experiment=experiment,
                sample=sample,
                model=model,
                params=params)

            """
            fig, axs = plt.subplots(1, 2)

            axs[0].imshow(sample)
            axs[0].set_title("Original image")
            axs[1].imshow(start_image)
            axs[1].set_title("Final perturbation")
            plt.show()

            
            success = decision_function(model, start_image[None], params)
            print("Is adversarial?", success)

            difference = np.abs(start_image - sample)
            print("Max difference:", np.max(difference))
            difference /= np.max(difference)


            plt.imshow(difference)
            plt.title("Scaled noise")
            plt.show()

            plt.imshow(start_image)
            plt.title("Adversarial image")
            plt.show()
            """

            init_dist = compute_distance(start_image, sample)

            boundary_img = binary_search(model, sample, start_image, params)
            boundary_dist = compute_distance(boundary_img, sample)

            eval_start[experiment].append(init_dist)
            eval_end[experiment].append(boundary_dist)

            if experiment == "fpar":
                #print("fpar used", query_counter.queries, "queries. Setting this as the cap for HSJA")
                print("fpar used", query_counter.queries, "queries.")
                if query_counter.queries > fpar_max_queries:
                    fpar_max_queries = query_counter.queries
                fpar_total_queries += query_counter.queries
                #query_counter.max_queries = query_counter.queries

            if experiment in CFG.USE_HSJA:
                final_img = run_hsja(model, sample, boundary_img)
                eval_end[experiment].append(compute_distance(final_img, sample))

            if experiment == "random":
                random_total_queries += query_counter.queries

            # Saving computed images to folder /results
            # result_image = np.concatenate([sample, np.zeros((sample.shape[0],8,3)), start_image, np.zeros((sample.shape[0],8,3)), bs_img, np.zeros((sample.shape[0],8,3)), final_img], axis = 1)
            # axs[j].imshow(result_image)
            # plt.imshow(result_image)
            # axs[j].title.set_text("Original image ({}) - After Init by {} ({}) - After Binary Search ({}) - After HSJA ({})".format(
            #    class_names[original_label],
            #    experiment, 
            #    class_names[np.argmax(model.predict(start_image))], 
            #    class_names[np.argmax(model.predict(bs_img))], 
            #    class_names[np.argmax(model.predict(final_img))]))
            # plt.show()

        if CFG.PRINT_ITERATION_SUMMARY:
            start_distances = []
            end_distances = []
            for k in experiments:
                start_distances.append(eval_start[k][-1])
                end_distances.append(eval_end[k][-1])
            best_exp[np.argmin(end_distances)] += 1

            print_iteration_summary(experiments, start_distances, end_distances, best_exp)

        if CFG.PRINT_ITERATION_MEDIANS_AND_MEANS:
            print_current_medians_and_averages(experiments, eval_start, eval_end)

        print("fpar max queries used:", fpar_max_queries)
        print("fpar mean queries used:", fpar_total_queries / (i+1))
        print("random mean queries used:", random_total_queries / (i+1))

    #plot_all_experiments(experiments)
    #plot_median(experiments)
    #fig.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
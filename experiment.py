import config as CFG

import numpy as np
import random
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)

import query_counter
from init_methods import get_start_image
from models import get_model
from datasets import get_dataset
from evaluation import start_eval_experiment, init_plotting, add_dist_queries, plot_all_experiments, plot_median, padding_queries, plot_success_rate

from utils.run_hsja import run_hsja
from utils.binary_search import binary_search
from utils.compute_distance import compute_distance
from utils.logging import setup_logging
from utils.printing import print_current_medians_and_averages, print_sample_progress, print_iteration_summary

from matplotlib import pyplot as plt
from utils.decision_function import decision_function

from utils.imagenet_human_readable_labels import get_classname, get_imagenet_classname

import json

from FCBSA import run_fcbsa
from HSJA.hsja import hsja

if __name__ == '__main__':

    model = get_model(CFG.MODEL)

    dataset, labels = get_dataset(
        CFG.DATASET,
        CFG.NUM_IMAGES,
        labels=CFG.LABELS if CFG.LABELS != None else None)

    experiments = CFG.EXPERIMENTS
    query_counter.init_queries()
    init_plotting(experiments)

    counter = 0

    for i, sample in enumerate(dataset):

        print_sample_progress(
            current_iteration=i+1,
            max_iteration=CFG.NUM_IMAGES,
            show_time=True)

        original_label = np.argmax(model.predict(sample))
        true_label = labels[i]

        #print("Predicted class:\t", original_label, get_imagenet_classname(original_label))
        #print("True class:\t\t", true_label, get_classname(true_label))

        if get_classname(true_label) != get_imagenet_classname(original_label):
            # Discard if original_label != label_validation_dataset
            print("Misclassified sample: contiune ... ")
            counter += 1
            continue
        
        
        params = {
                "original_label": original_label,
                "target_label": None,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "shape": sample.shape
        }

        # Check if defence classifies correctly
        defence_check = decision_function(model, [sample], params)

        if defence_check:
            print("Target model can't classifiy defence image")
            print("Skipping image ...")
            continue

        for j, experiment in enumerate(experiments):
            query_counter.reset_queries()
            start_eval_experiment(experiment)

            print("Experiment: ", experiment)

            if experiment == "hsja":
                perturbed = hsja(model, sample)
            else:
                perturbed = run_fcbsa(experiment, model, sample, params)

            #fig, axs = plt.subplots(1,2)
            #axs[0].imshow(sample)
            #axs[0].set_title(str(np.argmax(model.predict(sample))))
            #axs[1].imshow(perturbed)
            #axs[1].set_title(str(np.argmax(model.predict(perturbed))))
            #plt.show()
        with open('checkpoint/query_counter_eval_exp.json', 'w') as f:
            json.dump(query_counter.eval_exp, f)   

    print("Misclassified samples:", counter)

    if CFG.RUN_EVAL:
        padding_queries(experiments)
        plot_median(experiments)
        plot_success_rate(experiments)

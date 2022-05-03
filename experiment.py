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

import json

from FCBSA import run_fcbsa
from HSJA.hsja import hsja

if __name__ == '__main__':

    model = get_model(CFG.MODEL)

    

    dataset = get_dataset(CFG.DATASET, CFG.NUM_IMAGES)

    experiments = CFG.EXPERIMENTS
    query_counter.init_queries()
    init_plotting(experiments)

    for i, sample in enumerate(dataset):

        print_sample_progress(
            current_iteration=i+1,
            max_iteration=CFG.NUM_IMAGES,
            show_time=True)

        # TODO: Fetch original_label fra valid set
        original_label = np.argmax(model.predict(sample))

        # TODO: Discard if original_label != label_validation_dataset

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
                hsja(model, sample)
            else:
                run_fcbsa(experiment, model, sample, params)
        
        with open('checkpoint/query_counter_eval_exp.json', 'w') as f:
            json.dump(query_counter.eval_exp, f)
        
    if CFG.RUN_EVAL:
        padding_queries(experiments)
        #plot_all_experiments(experiments)
        plot_median(experiments)
        plot_success_rate(experiments)

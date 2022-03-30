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
import query_counter
from utils import binary_search, compute_distance
from utils.logging import setup_logging
from utils.printing import print_current_medians_and_averages, print_sample_progress

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

    for i, sample in enumerate(data):

        print_sample_progress(iter=i+1, max_iter=CFG.NUM_IMAGES, show_time=True)

        original_label = np.argmax(model.predict(sample))

        params = {
                "original_label": original_label,
                "target_label": None,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "shape": sample.shape
        }

        random_boundary_dist = 0

        for j, experiment in enumerate(experiments):

            # Reset query counter for each experiment
            query_counter.reset_queries()
            start_eval_experiment(experiment)

            start_image = get_start_image(
                experiment=experiment,
                sample=sample,
                model=model,
                params=params)

            init_dist = compute_distance(start_image, sample)

            boundary_img = binary_search(model, sample, start_image, params)
            boundary_dist = compute_distance(boundary_img, sample)

            eval_start[experiment].append(init_dist)
            eval_end[experiment].append(boundary_dist)

            if experiment in CFG.USE_HSJA:
                final_img = run_hsja(model, sample, boundary_img)
                eval_end[experiment].append(compute_distance(final_img, sample))

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

        print_current_medians_and_averages(experiment, eval_start, eval_end)
        
        start_distances = []
        end_distances = []
        for k in eval_start:
            start_distances.append(eval_start[k][-1])
            end_distances.append(eval_end[k][-1])

        print("Start distances:", start_distances)
        print("End distances:", end_distances)

        best_exp[np.argmin(end_distances)] += 1

        print("Experiments:", experiments)
        print("{}/{}:".format(str(i+1).rjust(len(str(CFG.NUM_IMAGES)), "0"), CFG.NUM_IMAGES), best_exp)

        
    #plot_all_experiments(experiments)
    #plot_median(experiments)

    print("Experiments:", experiments)
    print("Best:", best_exp)
    

        #fig.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
import query_counter as qc
from matplotlib import pyplot as plt
import numpy as np
import itertools
from datetime import datetime
from utils import decision_function
from imagenet_classes import class_names
import os

def init_plotting(experiements):
    for exp in experiements:
        qc.eval_exp[exp] = []

def start_eval_experiment(exp):
    qc.active_exp = exp
    qc.queries = 0
    qc.eval_exp[exp].append([])

def get_active_experiment():
    return qc.active_exp


def add_dist_queries(dist):
    # Append tuple (distance, #queries )
    qc.eval_exp[qc.active_exp][-1].append((dist, qc.queries))

def plot_all_experiments(path, experiements):
    plt.clf()
    list_of_colors = ["k", "r", "c", "m", "y", "b"]
    assert len(list_of_colors) >= len(experiements)

    #print(qc.eval_exp)

    for i, exp in enumerate(experiements):
        # line: points from image_pr_experiment
        for line in qc.eval_exp[exp]:
            extract_xy = list(zip(*line))
            plt.plot(extract_xy[1], extract_xy[0], "{}-".format(list_of_colors[i]))
    #plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    #plt.show()
    plt.savefig(path + "/plot_all_exps.png")


def plot_median(path, experiements):
    plt.clf()
    list_of_colors = ["k", "r", "c", "m", "y", "b"]
    assert len(list_of_colors) >= len(experiements)

    #print(qc.eval_exp)

    for i, exp in enumerate(experiements):

        num_iter = [len(i) for i in qc.eval_exp[exp]]
        x = []
        for itr in range(max(num_iter)):
            x.append(itr)
        #print("x: ", x)

        transposed = list(itertools.zip_longest(*qc.eval_exp[exp], fillvalue=None))

        y = []

        for itr in transposed:
            #print("itr: ", itr)
            # itr[] is all dist for iteration i
            y_itr = []
            for ys in itr:
                if isinstance(ys, tuple):
                    y_itr.append(ys[0])
            y_median = np.median(y_itr)
            y.append(y_median)
        
        #print("y: ", y)
        plt.plot(x, y, "{}-".format(list_of_colors[i]))
    #plt.savefig(path + "/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    plt.savefig(path + "/plot_of_medians.png")
    #plt.show()

def save_experiment_image(model, path, iteration, original_img, init_pert, boundary_pert, final_pert):
    plt.clf()
    result_image = np.concatenate([original_img, 
    np.zeros((original_img.shape[0],8,3)), 
    init_pert, 
    np.zeros((original_img.shape[0],8,3)), 
    boundary_pert, 
    np.zeros((original_img.shape[0],8,3)), 
    final_pert], 
    axis = 1)

    plt.imshow(result_image)
    plt.title(
        "original: ({}) - init_pert: ({}) - boundary_pert: ({}) - final_pert ({})".format(
        class_names[np.argmax(model.predict(original_img))],
        class_names[np.argmax(model.predict(init_pert))], 
        class_names[np.argmax(model.predict(boundary_pert))], 
        class_names[np.argmax(model.predict(final_pert))])
        )
    plt.savefig(path + "/attack_process_imgs/" + "iteration_"+ str(iteration) + "_" + qc.active_exp)
        

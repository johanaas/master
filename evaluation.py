import query_counter as qc
from matplotlib import pyplot as plt
import numpy as np
import itertools
from datetime import datetime

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

def plot_all_experiments(experiements):
    list_of_colors = ["k", "r", "c", "m", "y", "b"]
    assert len(list_of_colors) >= len(experiements)

    #print(qc.eval_exp)

    for i, exp in enumerate(experiements):
        # line: points from image_pr_experiment
        for line in qc.eval_exp[exp]:
            extract_xy = list(zip(*line))
            plt.plot(extract_xy[1], extract_xy[0], "{}-".format(list_of_colors[i]))
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    #plt.show()


def plot_median(experiements):

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
    plt.legend(experiements)
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    #plt.show()


def padding_queries(experiements):

    padding = 1500

    for i, exp in enumerate(experiements):
        for img in qc.eval_exp[exp]: # img: [(), (), ()]
            end_dist = img[-1][0]
            end_query = img[-1][1]
            for j in range(end_query + 1, padding):
                img.append((end_dist, j))

def plot_success_rate(experiements):

    end_dist = {}

    list_of_colors = ["k", "r", "c", "m", "y", "b"]
    assert len(list_of_colors) >= len(experiements)

    for i, exp in enumerate(experiements):
        end_dist[exp] = []
        for img in qc.eval_exp[exp]: # img: [(), (), ()]
            end_dist[exp].append(img[-1][0])

    
    for j, exp in enumerate(experiements):
        x = []
        y = []
        for i in np.arange(0, 30.1, 0.1):
            x.append(i)
            s_rate = 1 - len([element for element in end_dist[exp] if element > i]) / len(end_dist[exp])
            y.append(s_rate)
        plt.plot(x, y, "{}-".format(list_of_colors[j]))
    plt.legend(experiements)
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    #plt.show()
import query_counter as qc
from matplotlib import pyplot as plt
import numpy as np
import itertools
from datetime import datetime
import pickle
import config as CFG

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

def load_from_file(experiements):
    with open('checkpoint/query_counter_eval_exp.pickle', 'rb') as handle:
        qc.eval_exp = pickle.load(handle)

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
    #plt.yscale("log")
    plt.xlabel("Number of Queries")
    plt.ylabel(r'$l_2$ Distance')
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    plt.show()


def plot_queries_used():
    end_queries = []
    
    for img in qc.eval_exp["dyn-fcbsa2"]: # img: [(), (), ()]
        end_dist = img[-1][0]
        end_query = img[-1][1]
        end_queries.append(end_query)


    # Find max query and round up to closest hundred
    max_query = np.max(end_queries)
    print(max_query)
    rounded_max_query = int(np.ceil(max_query / 100)) * 100

    
    plt.hist(end_queries, bins=int(rounded_max_query/100), range=(0,rounded_max_query))
    plt.show()

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
    plt.ylabel("Success Rate")
    plt.xlabel(r'$l_2$ Distance')
    plt.grid()
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    plt.show()


if __name__ == '__main__':
    qc.init_queries()
    experiments = CFG.EXPERIMENTS
    load_from_file(experiments)

    # Call your evalatuon methods
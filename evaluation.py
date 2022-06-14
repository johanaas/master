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

def load_from_file(experiements, experiment_id):
    with open(experiment_id, 'rb') as handle:
        qc.eval_exp = pickle.load(handle)

def plot_median(experiements):

    list_of_colors = ["c", "m", "k", "r", "g", "y", "b"]
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
        print(exp)
        if exp[0] == "M":
            if exp == "MASSA":
                plt.plot(x, y, "c--")
            else:
                plt.plot(x, y, "c-")
        else:
            if exp == "HSJA":
                plt.plot(x, y, "m--")
            else:
                plt.plot(x, y, "m-")
        
        #plt.plot(x, y, "{}-".format(list_of_colors[i]))
    legends = [] #["MASSA", "HSJA", "MASSA-DEFENCE", "HSJA-DEFENCE"]
    for key in qc.eval_exp:
        legends.append(key)
    plt.legend(legends)
    #plt.yscale("log")
    plt.xlabel("Number of Queries")
    plt.ylabel(r'$l_2$ Distance')
    plt.ylim(bottom=0)
    plt.style.use('seaborn-whitegrid')
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    plt.show()


def plot_queries_used():
    end_queries = []
    
    for img in qc.eval_exp["dyn-fcbsa2"]: # img: [(), (), ()]
        end_dist = img[-1][0]
        end_query = img[-1][1]
        end_queries.append(end_query)
    print(len(end_queries))
    print("Queries < 250: ", sum( 1 for q in end_queries if q < 250 ) / len(end_queries))
    print("Queries < 500: ", sum( 1 for q in end_queries if q < 500 ) / len(end_queries))
    print("Queries < 750: ", sum( 1 for q in end_queries if q < 750 ) / len(end_queries))
    print("Queries < 1000: ", sum( 1 for q in end_queries if q < 1000 ) / len(end_queries))


    # Find max query and round up to closest hundred
    max_query = np.max(end_queries)
    print(max_query)
    rounded_max_query = int(np.ceil(max_query / 100)) * 100

    plt.style.use('seaborn-whitegrid')
    plt.hist(end_queries, bins=int(rounded_max_query/100), facecolor = "#2ab0ff", edgecolor = "#169acf", linewidth = 0.5,  range=(0,rounded_max_query))
    plt.ylabel("Number of Images")
    plt.xlabel("Number of Queries")
    plt.xticks(range(0,rounded_max_query + 100, 100))
    
    plt.show()

def padding_queries(experiements):

    padding = 1500

    for i, exp in enumerate(experiements):
        for img in qc.eval_exp[exp]: # img: [(), (), ()]
            end_dist = img[-1][0]
            end_query = img[-1][1]
            for j in range(end_query + 1, padding):
                img.append((end_dist, j))

def list_distance(experiements, dist):
    
    dist_exp = {}

    for i, exp in enumerate(experiements):
        dist_exp[exp] = []
        for img in qc.eval_exp[exp]: # img: [(), (), ()]
            dist_exp[exp].append(img[dist - 1][0])

        # print avg and median
        print(exp, " - ", dist)
        print("Median: ", np.median(dist_exp[exp]))
        print("Average: ", np.mean(dist_exp[exp]))


def cap_queries(experiements, cap):

    eval_exp = {}

    for i, exp in enumerate(experiements):
        eval_exp[exp] = []
        for img in qc.eval_exp[exp]: # img: [(), (), ()]
            eval_exp[exp].append(img[:cap])

    qc.eval_exp = eval_exp

    #for k in eval_exp["hsja"]:
    #    #print(list(duplicates(eval_exp["hsja"][0])))
    #    duplicates = [number for number in k if k.count(number) > 1]
    #    unique_duplicates = list(set(duplicates))
    #    print(unique_duplicates)

def plot_success_rate(experiements):


    caps = [1000, 750, 500, 250]

    end_dist = {}


    for i, exp in enumerate(experiements):
        #new_eval_exp[exp + "-1000"] = qc.eval_exp[exp]
        for cap in caps:
            end_dist[exp + "-" + str(cap)] = []
            for img in qc.eval_exp[exp]: # img: [(), (), ()]
                end_dist[exp + "-" + str(cap)].append(img[cap][0])

    #print(new_eval_exp["dyn-fcbsa2-250"][0])

    #print(new_eval_exp.keys)
    #for key in new_eval_exp:
    #    print(key)

    

    list_of_colors = ["c", "m", "k", "r", "g", "y", "b"]
    assert len(list_of_colors) >= len(experiements)

    #for i, exp in enumerate(experiements):
    #    end_dist[exp] = []
    #    
    #    for img in qc.eval_exp[exp]: # img: [(), (), ()]
    #        
    #        end_dist[exp].append(img[-1][0])
    end_dist_keys = []
    for key in end_dist:
        end_dist_keys.append(key)
    plt.style.use('seaborn-whitegrid')
    for j, exp in enumerate(end_dist_keys):
        x = []
        y = []
        for i in np.arange(0, 30.1, 0.1):
            x.append(i)
            s_rate = 1 - len([element for element in end_dist[exp] if element > i]) / len(end_dist[exp])
            y.append(s_rate)
        
        if exp.split("-")[0] == "dyn":
            if exp.split("-")[-1] == "250":
                plt.plot(x, y, "c:")
            if exp.split("-")[-1] == "500":
                plt.plot(x, y, "c-.")
            if exp.split("-")[-1] == "750":
                plt.plot(x, y, "c--")
            if exp.split("-")[-1] == "1000":
                plt.plot(x, y, "c-")
        else:
            if exp.split("-")[-1] == "250":
                plt.plot(x, y, "m:")
            if exp.split("-")[-1] == "500":
                plt.plot(x, y, "m-.")
            if exp.split("-")[-1] == "750":
                plt.plot(x, y, "m--")
            if exp.split("-")[-1] == "1000":
                plt.plot(x, y, "m-")
    legends = []
    for key in end_dist_keys:
        if key.split("-")[0] == "dyn":
            legends.append(key.replace("dyn-fcbsa2", "MASSA"))
        else:
            legends.append(key.replace("hsja", "HSJA"))
        

    plt.legend(legends)
    plt.ylabel("Success Rate")
    plt.xlabel(r'$l_2$ Distance')
    #plt.style.use('seaborn-whitegrid')
    plt.savefig("results/{}.png".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    plt.show()


if __name__ == '__main__':
    qc.init_queries()
    experiments = CFG.EXPERIMENTS

    experiment_id = "checkpoint/{}_imgs_{}_model_{}_defence_{}_seed_{}.pickle".format(
        "_".join(CFG.EXPERIMENTS).replace(".", "_"),
        CFG.CAP_IMGS,
        CFG.MODEL,
        CFG.DEFENCE,
        CFG.SEED
    )

    load_from_file(experiments, experiment_id)
    plot_queries_used()
    padding_queries(experiments)
    list_distance(experiments, 1000)
    list_distance(experiments, 750)
    list_distance(experiments, 500)
    list_distance(experiments, 250)
    plot_success_rate(experiments)

    cap_queries(experiments, 1000)
    
    # Call your evalatuon methods

    #eval_dict = {}
    #for key in qc.eval_exp:
    #    if key == "dyn-fcbsa2":
    #        eval_dict["MASSA-DEFENCE"] = qc.eval_exp[key]
    #    else:
    #        eval_dict["HSJA-DEFENCE"] = qc.eval_exp[key]
    #qc.eval_exp = eval_dict
    #experiment_id = "finished_experiments/{}_imgs_{}_model_resnet50_defence_None_seed_{}.pickle".format(
    #    "_".join(CFG.EXPERIMENTS).replace(".", "_"),
    #    CFG.CAP_IMGS,
    #    #CFG.MODEL,
    #    CFG.SEED
    #)
    #with open(experiment_id, 'rb') as handle:
    #    eval_dict = pickle.load(handle)
    #    for key in eval_dict:
    #        if key == "dyn-fcbsa2":
    #            qc.eval_exp["MASSA"] = eval_dict[key]
    #        else:
    #            qc.eval_exp["HSJA"] = eval_dict[key]

    #for key in qc.eval_exp:
    #    print(key)
    #experiments = ["MASSA", "HSJA", "MASSA-DEFENCE", "HSJA-DEFENCE"]
    #padding_queries(experiments)
    #
    #list_distance(experiments, 1000)
    #list_distance(experiments, 750)
    #list_distance(experiments, 500)
    #list_distance(experiments, 250)

    #cap_queries(experiments, 1000)
    #plot_median(experiments)
    
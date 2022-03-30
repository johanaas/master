from datetime import datetime
import numpy as np

def print_sample_progress(current_iteration, max_iteration, show_time=False):
    print("-----------------------------------------------")
    print("Attacking sample nr {} / {}".format(current_iteration, max_iteration))

    if show_time:
        print("Time: ", datetime.now())

def print_current_medians_and_averages(experiments, starts, ends, width=40):

    print("Current status".center(width, "-"))
    
    for e in experiments:
        print(e, "Start \tMedian: {}\tAverage: {}".format(np.median(starts[e]), np.mean(starts[e])))
        print(e, "End \tMedian: {}\tAverage: {}".format(np.median(ends[e]), np.mean(ends[e])))
        
    print("-"*width)
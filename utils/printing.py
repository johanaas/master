from datetime import datetime
import numpy as np

def print_sample_progress(current_iteration, max_iteration, show_time=False):
    print("\n\n-----------------------------------------------")
    print("Attacking sample nr {} / {}".format(current_iteration, max_iteration))

    if show_time:
        print("Time: ", datetime.now())

def print_current_medians_and_averages(experiments, starts, ends, width=40):

    print("Current status".center(width, "-"))
    
    for e in experiments:
        print(e, "Start \tMedian: {}\tAverage: {}".format(np.median(starts[e]), np.mean(starts[e])))
        print(e, "End \tMedian: {}\tAverage: {}".format(np.median(ends[e]), np.mean(ends[e])))
        print()
        
    print("-"*width)

def print_iteration_summary(experiments, starts, ends, best, width=20):
    print("Experiments".ljust(width), ":", experiments)
    print("Start distances".ljust(width), ":", starts)
    print("End distances".ljust(width), ":", ends)
    print("Best counter".ljust(width), ":", best)
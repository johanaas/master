import json
from evaluation import start_eval_experiment, init_plotting, add_dist_queries, plot_all_experiments, plot_median, padding_queries, plot_success_rate
import query_counter
 
# Opening JSON file
f = open('checkpoint/query_counter_eval_exp.json')
 
# returns JSON object as
# a dictionary
query_counter.init_queries()
query_counter.eval_exp = json.load(f)
 
# Closing file
f.close()

print(query_counter.eval_exp["dyn-fcbsa2"])

EXPERIMENTS = ["dyn-fcbsa2"]#["dyn-fcbsa0.5", "dyn-fcbsa1", "dyn-fcbsa1.5", "dyn-fcbsa2", "dyn-fcbsa2.5", "dyn-fcbsa3"]

padding_queries(EXPERIMENTS)
plot_all_experiments(EXPERIMENTS)
plot_median(EXPERIMENTS)
plot_success_rate(EXPERIMENTS)
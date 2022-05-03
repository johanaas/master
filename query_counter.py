import config as CFG

def init_queries():
    global queries
    queries = 0
    global max_queries
    max_queries = CFG.MAX_NUM_QUERIES
    global eval_exp
    eval_exp = {}
    global active_exp
    active_exp = ""
    global prev_dist
    prev_dist = -1
    global end_dist
    end_dist = []

def reset_queries():
    global queries
    queries = 0
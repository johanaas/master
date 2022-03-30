from HSJA.hsja import hsja

def run_hsja(model, sample, sample_perturbed):
    perturbed = hsja(
        model, 
        sample,
        sample_perturbed,
        clip_max = 1.0,
        clip_min = 0.0, 
        constraint = "l2",
        num_iterations = 25,
        gamma = 1.0,
        target_label = None,
        target_image = None,
        stepsize_search = "geometric_progression",
        max_num_evals = 1e4,
        init_num_evals = 100)
    return perturbed
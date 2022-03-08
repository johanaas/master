""" Main config file for running the experiments """

# Main seed. If not None it will be used as a seed for
# numpy and pythons random module
SEED = 69

# Experiments to run. One experiment consists of using that
# specific method to generate a starting point, and then
# performing HSJA for MAX_NUM_QUERIES.
# Supported values: ["random", "par"]
EXPERIMENTS = ["par"]


# Max number of model queries allowed. Attack breaks if
# this value is exceeded
MAX_NUM_QUERIES = 250


# The number of images to load from the dataset
# An adversarial example is generated for each image
# per experiment
NUM_IMAGES = 1
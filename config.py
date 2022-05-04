""" Main config file for running the experiments """

# Main seed. If not None it will be used as a seed for
# numpy and pythons random module
SEED = 13

# Experiments to run. One experiment consists of using that
# specific method to generate a starting point, and then
# performing HSJA for MAX_NUM_QUERIES.
# Supported values: ["random", "par"]
EXPERIMENTS = ["dyn-fcbsa2.5"]#["fcbsa", "dyn-fcbsa0.5", "dyn-fcbsa1", "dyn-fcbsa1.5", "dyn-fcbsa2", "dyn-fcbsa2.5", "dyn-fcbsa3"]


# List of experiments that use HSJA. Name have to match the
# names in EXPERIMENTS
USE_HSJA = ["random"]


# Max number of model queries allowed. Attack breaks if
# this value is exceeded
MAX_NUM_QUERIES = 1000


# The number of images to load from the dataset
# An adversarial example is generated for each image
# per experiment
NUM_IMAGES = 50


# The model to use
# Supported values: "resnet50 | resnet101"
MODEL = "resnet50"


# Choose if or which defence to use
# Supported values: "None | JPEG"
DEFENCE = None


# The dataset to use
# Supported values: "imagenet | cifar10 | cifar100 | mnist"
DATASET = "imagenet"

# The path to the manually downloaded imagenet dataset
IMAGENET_PATH = r"C:\Users\johanaas\Documents\HSJA\ILSVRC2012\test"


# If not None all print statements are written to a logfile
# The directory has to be manually created, and the logfile
# is automatically created inside the directory
LOG_DIR = "logs"


# Collect distance and queries in order to plot graphs
RUN_EVAL = True

# Print median and average distances for each iteration
PRINT_ITERATION_MEDIANS_AND_MEANS = True

# Print start and end distances for each experiment in
# each iteration
PRINT_ITERATION_SUMMARY = True
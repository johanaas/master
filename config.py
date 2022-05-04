""" Main config file for running the experiments """

# Main seed. If not None it will be used as a seed for
# numpy and pythons random module
SEED = 694

# Experiments to run. One experiment consists of using that
# specific method to generate a starting point, and then
# performing HSJA for MAX_NUM_QUERIES.
# Supported values: ["random", "par"]
EXPERIMENTS = ["circular_fpar"]


# List of experiments that use HSJA. Name have to match the
# names in EXPERIMENTS
USE_HSJA = []


# Max number of model queries allowed. Attack breaks if
# this value is exceeded
MAX_NUM_QUERIES = 1000


# The number of images to load from the dataset
# An adversarial example is generated for each image
# per experiment
NUM_IMAGES = 1


# The model to use
# Supported values: "resnet50 | resnet101"
MODEL = "resnet50"


# The dataset to use
# Supported values: "imagenet | cifar10 | cifar100 | mnist"
DATASET = "imagenet"

# The path to the manually downloaded imagenet dataset
IMAGENET_PATH = r"C:\Users\kamidtli\dev\ILSVRC2012_img_test\test"


# If not None all print statements are written to a logfile
# The directory has to be manually created, and the logfile
# is automatically created inside the directory
LOG_DIR = None #"logs"


# Collect distance and queries in order to plot graphs
RUN_EVAL = False

# Print median and average distances for each iteration
PRINT_ITERATION_MEDIANS_AND_MEANS = True

# Print start and end distances for each experiment in
# each iteration
PRINT_ITERATION_SUMMARY = True
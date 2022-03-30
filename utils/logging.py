import config as CFG
import sys

def setup_logging():

    logfile_path = "{}/{}_cap{}_{}imgs_{}_{}.txt".format(
        CFG.LOG_DIR,
        "_".join(CFG.EXPERIMENTS),
        CFG.MAX_NUM_QUERIES,
        CFG.NUM_IMAGES,
        CFG.DATASET,
        CFG.MODEL
      )
    sys.stdout = Logger(sys.stdout, logfile_path)

class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if isinstance(f, str):
                file = open(f, 'a')
                file.write(obj)
                file.close()
            else:
                f.write(obj)

    def flush(self):
        pass
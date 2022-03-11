import cv2
import numpy as np
import random
import os
import config as CFG


def load_imagenet(num_images):
    path = CFG.IMAGENET_PATH
    random.seed(69)
    random_filenames = random.sample([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ], num_images)

    random_images = []
    for x in random_filenames:
        random_images.append(process_image(os.path.join(path, x)))

    return random_images

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def process_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    if img.shape[0] < img.shape[1]:
        # Hoyde er minst
        new_height = 256
        new_width = round((img.shape[1] * new_height) / img.shape[0])

        # dsize
        dsize = (new_width, new_height)

        # resize image
        resized = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)

        # Do centercrop for 224x224
        output = crop_center(resized, 224, 224)

    else:
        new_width = 256
        new_height = round((img.shape[0] * new_width) / img.shape[1])

        # dsize
        dsize = (new_width, new_height)

        # resize image
        resized = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)

        output = crop_center(resized, 224, 224)

    return output / 255
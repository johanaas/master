from cProfile import label
import cv2
import numpy as np
import random
import os
import config as CFG
from utils.imagenet_human_readable_labels import get_classname

from mat4py import loadmat

import sys

def load_imagenet(num_images, labels_file_path=None):
    path = CFG.IMAGENET_PATH
    """
    random_filenames = [
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ][:10]
    """
    random_filenames = random.sample([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ], num_images)

    random_images = []
    image_indices = []
    for x in random_filenames:
        random_images.append(process_image(os.path.join(path, x)))
        
        if labels_file_path != None:
            image_num = int(x.split(".")[0].split("_")[-1])
            image_indices.append(image_num - 1)

    labels = []    
    if labels_file_path != None:
        with open(labels_file_path) as file:
            for index, line in enumerate(file):
                if index in image_indices:
                    label = int(line)
                    labels.append(label)
                    #print(label, ":", get_classname(label - 1))
    print(image_indices)
    print(len(image_indices))
    print(labels)
    print(len(labels))
    return random_images, labels

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
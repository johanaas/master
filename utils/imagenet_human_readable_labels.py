import json
from mat4py import loadmat

def get_classname(class_num=0):
    data = loadmat(r"C:\Users\johanaas\Documents\new_HSJA\meta.mat")
    return data["synsets"]["words"][class_num - 1]

def get_imagenet_classname(class_num=0):
    with open(r"C:\Users\johanaas\Documents\new_HSJA\ILSVRC2012_human_readable.txt") as file:
        json_str = file.read()
        json_obj = json.loads(json_str)
        return json_obj[str(class_num)]
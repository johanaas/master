import keras
import numpy as np
import query_counter
import torch
import torchvision
from collections import OrderedDict
from torchvision import transforms


class adv_ResnetModel50():
    def __init__(self):

        self.model = torchvision.models.resnet50(pretrained=False)
        
        #print(torch.load('pretrained_models/imagenet_l2_3_0.pt', map_location="cpu")["model"].keys())
        state_dict = torch.load('pretrained_models/imagenet_l2_3_0.pt', map_location="cpu")["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "attacker" in k:
                if "normalize" in k:
                    boi = "boi"
                else:
                    name = k[22:]
                    new_state_dict[name] = v
        #print(new_state_dict.keys())
        self.model.load_state_dict(new_state_dict)
        #= torch.load('pretrained_models/imagenet_l2_3_0.pt', map_location ='cpu')
        self.model.eval()

    def predict(self, x, verbose=0, batch_size = 500, logits = False):
        x = np.array(x) #* 255
        
        #preprocess = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        x -= mean
        x/= std
        #print(x.shape)
        
        #print(x.shape)
        #x = preprocess(x)
        #print(x.shape)
        #x = np.moveaxis(x, -1, 0)
        if len(x.shape) == 3:
            _x = np.expand_dims(x, 0) 
        else:
            _x = x
        #_x = torch.unsqueeze(x, 0)
        #print(_x.shape)
        _x = np.moveaxis(_x, -1, 1)
        #_x = preprocess(_x)
        
        _x = torch.FloatTensor(_x)
        # Prepross input
        prob = self.model.forward(_x)
        return prob.detach().numpy()
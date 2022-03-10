import keras
import numpy as np
import settings

class ResnetModel50():
    def __init__(self):
        self.model = keras.applications.resnet50.ResNet50(
            include_top=True, 
            weights='imagenet', 
            input_tensor=None, 
            input_shape=None, 
            pooling=None, 
            classes=1000)

    def predict(self, x, verbose=0, batch_size = 500, logits = False):
        x = np.array(x) * 255
        if len(x.shape) == 3:
            _x = np.expand_dims(x, 0) 
        else:
            _x = x
        prob = self.model.predict(_x, batch_size = batch_size)
        settings.queries += len(prob)
        return prob
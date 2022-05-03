import keras
import numpy as np
import query_counter

class MobileNet():
    def __init__(self):
        self.model = keras.applications.mobilenet.MobileNet(
            input_shape=None,
            alpha=1.0,
            depth_multiplier=1,
            dropout=0.001,
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            classes=1000,
        )

    def predict(self, x, verbose=0, batch_size = 500, logits = False):
        x = np.array(x) * 255
        if len(x.shape) == 3:
            _x = np.expand_dims(x, 0) 
        else:
            _x = x
        prob = self.model.predict(_x, batch_size = batch_size)
        #query_counter.queries += len(prob)
        return prob
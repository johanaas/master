import keras

def load_cifar10(num_images=1):
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    data = x_test.astype('float32')/255

    if num_images > len(data):
      raise ValueError("Cannot get", num_images, "from dataset with", len(data), "images")

    return data[:num_images]

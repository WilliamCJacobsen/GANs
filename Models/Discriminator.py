from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Reshape, Conv2D, LeakyReLU, Flatten,AveragePooling2D
import numpy as np

#TODO: Print the loss and accuracy under trianing
class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def create_discriminator(self, optimizer):
        print("create the discriminator...")
        model = Sequential([
            Conv2D(32, (3,3), strides = (2,2), input_shape=self.input_shape, padding = "same"),
            LeakyReLU(),
            AveragePooling2D((2,2)),
            BatchNormalization(momentum=0.8),
            Conv2D(64, (3,3), strides = (2,2), padding = "same"),
            AveragePooling2D((2,2)),
            LeakyReLU(),
            Flatten(),
            Dense(1)
        ])
        self.model = model
        print(model.summary())
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        model.trainable = False
        return model

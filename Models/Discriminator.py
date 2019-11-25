from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Reshape, Conv2D, LeakyReLU, Flatten
import numpy as np


class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def create_discriminator(self):
        model = Sequential([
            Conv2D(32, (3,3), strides = (2,2), input_shape=self.input_shape, padding = "same"),
            LeakyReLU(),
            BatchNormalization(momentum=0.8),
            Conv2D(64, (3,3), strides = (2,2), padding = "same"),
            LeakyReLU(),
            Flatten(),
            Dense(1)
        ])

        print(model.summary())
        return model

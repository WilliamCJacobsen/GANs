from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Reshape, UpSampling2D, Conv2D
import numpy as np


class Generator:
    def __init__(self, generator_shape: int, noise: np.array, output_shape: int):
        self.shape = generator_shape
        self.noise = noise
        self.output_shape = output_shape


    def create_generator(self):
        model = Sequential([
                Dense(64 * 3 * 3 , input_shape=self.noise, activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                Reshape(64 * 3 * 3),
                UpSampling2D(),
                Conv2D(32, (3,3), padding='same', activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                Conv2D(64, (3,3), padding='same',activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                Dense(np.prod(self.img_shape), activation='tanh'),
                Reshape(self.output_shape)
        ])
        print(model.summary())
        return model

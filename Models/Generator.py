from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Reshape, UpSampling2D, Conv2D,ConvTranspose2d
import numpy as np


class Generator:
    def __init__(self, noise_shape: tuple, output_shape: tuple):
        self.shape = noise_shape
        self.noise = noise
        self.output_shape = output_shape
        self.model = None

    def create_generator(self, optimizer):
        model = Sequential([
                Dense(64 * 3 * 3 , input_shape=self.noise, activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                Reshape(64 * 3 * 3),
                UpSampling2D(),
                ConvTranspose2d(32, (3,3), padding='same', activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                ConvTranspose2d(64, (3,3), padding='same',activation="leaky-relu"),
                BatchNormalization(momentum=0.8),
                Dense(np.prod(self.img_shape), activation='tanh'),
                Reshape(self.output_shape)
        ])
        print(model.summary())
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.model = model
        return model

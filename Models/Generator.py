from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization, Reshape, UpSampling2D, Conv2D,Conv2DTranspose, LeakyReLU
import numpy as np


class Generator:
    def __init__(self, noise_shape: tuple, image_shape: tuple):
        self.noise_shape = noise_shape
        self.image_shape = image_shape
        self.model = None

    def create_generator(self, optimizer):
        print("create the generator...")
        model = Sequential([
                Dense(64 * 3 * 3 , input_shape=self.noise_shape),
                LeakyReLU(),
                BatchNormalization(momentum=0.8),
                Reshape((7,7,256)),
                UpSampling2D(),
                Conv2DTranspose(32, (3,3), padding='same'),
                LeakyReLU(),
                BatchNormalization(momentum=0.8),
                Conv2DTranspose(64, (3,3), padding='same'),
                LeakyReLU(),
                BatchNormalization(momentum=0.8),
                Dense(np.prod(self.image_shape), activation='tanh'),
                Reshape(self.image_shape)
        ])
        print(model.summary())
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.model = model
        return model

from keras.models import Sequential, Model
from keras.layers import Dense, Activation,BatchNormalization, Reshape, UpSampling2D, Conv2D,Conv2DTranspose, LeakyReLU, Input, Flatten
import numpy as np

class Generator:
    def __init__(self, noise_shape: tuple, image_shape: tuple):
        self.noise_shape = noise_shape
        self.image_shape = image_shape
        self.model = None


#TODO: make this a Convnet.
    def create_generator(self, optimizer):
        print("create the generator...")
        model = Sequential([
                Dense(32*32 , input_dim=self.noise_shape),
                BatchNormalization(momentum=0.8),
                Reshape(target_shape=(32,32,1)),
                Conv2D(32,(3,3), padding='same'),
                BatchNormalization(momentum=0.8),
                LeakyReLU(),
                Conv2DTranspose(filters = 64, kernel_size=  (3,3), strides = (2,2), padding= 'same'),
                LeakyReLU(),
                Conv2D(32,(5,5), padding='same'),
                BatchNormalization(momentum=0.8),
                LeakyReLU(),
                Flatten(),
                Dense(np.prod(self.image_shape), activation='tanh'),
                Reshape(self.image_shape)
        ])
        print(model.summary())
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        noise = Input(shape = (self.noise_shape,))
        img = model(noise)

        self.model = Model(noise, img)
        return self.model

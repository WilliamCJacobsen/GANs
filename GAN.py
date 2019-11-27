from Models.Discriminator import Discriminator
from Models.Generator import Generator
#keras imports
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
#tool imports
import numpy as np
import matplotlib.pyplot as plt
import time

class GAN():
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = None


    def load_data(self):
        # gjør data om til normalisert mellom -1 og 1
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5)/127.5
        return (x_train, y_train, x_test, y_test)


# adding the optimizer for both of the models
    def create_gan(self):
        gan_input = Input(shape=(100,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam())
        gan.summary()
        self.gan = gan


    def plot_generated_image(self, epoch, figure_size = (28,28)):
        noise = np.random.normal(loc=0, scale=1, size=[1,100])
        generate_image = self.generator.predict(noise).reshape(figure_size)
        plt.imshow(generate_image)
        plt.savefig(f'images/image{time.time()}.png')


    def training(self, batch_size = 32, epoch = 10000):
        (X_train, y_train, X_test, y_test) = self.load_data()
        y_labels = np.ones(batch_size)
        self.create_gan()

        for e in range(epoch):
            print(f"epoch # {e}")
            for batch in range(batch_size):
                noise = np.random.normal(0,1,[batch_size, 100])
                generated_images = self.generator.predict(noise)
                # Get a random set of  real images
                image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)].reshape(batch_size,28,28,1)

                X = np.append(generated_images, image_batch).reshape(batch_size*2,28,28,1)

                y_discriminator = np.zeros(2*batch_size)
                y_discriminator[:batch_size] = 0.9
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_discriminator)
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise,y_labels)
        
            if e % 1000 == 0:
                generated_images = self.generator.predict(noise)
                # Get a random set of  real images
                image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)].reshape(batch_size,28,28,1)
                
                X = np.append(generated_images, image_batch).reshape(batch_size*2,28,28,1)
                loss = self.discriminator.evaluate(X,y_discriminator)
                print(f"loss : {loss}")
                self.plot_generated_image(e)


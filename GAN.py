from Models.Discriminator import Discriminator
from Models.Generator import Generator
#keras imports
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
#tool imports
import numpy as np
import matplotlib.pyplot as plt

size_of_image = (28,28,1)
noise_shape = (64,)

discriminator = Discriminator(input_shape=size_of_image)
generator = Generator(noise_shape= noise_shape, output_shape= size_of_image)

# create the models of discriminator and generator
dis_model = discriminator.create_discriminator()
gen_model = generator.create_generator()

# adding the optimizer for both of the models
def create_gan(discriminator, generator):
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    gan.summary()
    return gan

def plot_generated_image(epoch, generator, examples =10, dim=(10,10), figure_size = (28,28,1)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generate_image = generator.predict(noise).reshape(100,28,28,1)
    plt.figure(figuresize = figure_size)
    for i in range(generate_image.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generate_image[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()




from Models.Discriminator import Discriminator
from Models.Generator import Generator

from keras.optimizers import Adam
from GAN import GAN



if __name__ == "__main__":
    noise_shape = 100
    generator_output = (28,28,1)

    generator = Generator(noise_shape,  generator_output).create_generator(Adam(0.01))
    discriminator = Discriminator(generator_output).create_discriminator(Adam(0.01))

    gan = GAN(generator = generator, discriminator = discriminator)

    gan.create_gan()
    print("starting to train")
    gan.training()


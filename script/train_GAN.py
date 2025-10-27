import Preprocessing_GAN
import GAN
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

#

def train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, train_nuclei=None, train_cells=None, val_nuclei=None, val_cells=None, validation_interval=1000, save_interval=1000):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, train_nuclei.shape[0], batch_size)
        nuclei_batch = train_nuclei[idx]
        cell_batch = train_cells[idx]
        noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))
        generated_images = generator.predict([nuclei_batch, noise])

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(cell_batch, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch([nuclei_batch, noise], valid_labels)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss}] [G loss: {g_loss}]")


        if epoch % save_interval == 0:
            # Save the models at regular intervals
            generator.save(f'../models/generator_epoch_{epoch}_g_loss_{g_loss}.h5')
            discriminator.save(f'../models/discriminator_epoch_{epoch}_d_loss_{d_loss}.h5')

    # Save the final models
    generator.save('../models/generator_final.h5')
    discriminator.save('../models/discriminator_final.h5')

# --- Main Execution ---

# Load and prepare data
nuclei_folder = '../data/2_norm_images/nuclei/'
cell_folder = '../data/2_norm_images/cell/'
nuclei_patches, cell_patches = Preprocessing_GAN.prepare_data(nuclei_folder, cell_folder)

# Split data into training and validation sets
train_nuclei, val_nuclei, train_cells, val_cells = Preprocessing_GAN.split_data(nuclei_patches, cell_patches)

# Initialize GAN components
generator = GAN.build_generator()
discriminator = GAN.build_discriminator()
gan = GAN.build_gan(generator, discriminator)

# Compile models
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Train GAN
train_gan(generator, discriminator, gan, epochs=10000, batch_size=8, train_nuclei=train_nuclei, train_cells=train_cells, val_nuclei=val_nuclei, val_cells=val_cells, validation_interval=1000)

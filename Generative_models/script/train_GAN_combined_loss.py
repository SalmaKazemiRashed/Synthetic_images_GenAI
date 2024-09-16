import Preprocessing_GAN
import GAN
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import GAN_combined_loss


def train_gan(generator, discriminator, gan, epochs, batch_size, train_nuclei, train_cells, val_nuclei=None, val_cells=None, validation_interval=10, save_interval=10):
    for epoch in range(epochs):
        indices = np.arange(len(train_nuclei))
        np.random.shuffle(indices)
        train_nuclei = train_nuclei[indices]
        train_cells = train_cells[indices]

        for i in range(0, len(train_nuclei), batch_size):
            batch_nuclei = train_nuclei[i:i + batch_size]
            batch_cells = train_cells[i:i + batch_size]
            batch_noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))

            generated_images = generator.predict([batch_nuclei, batch_noise])

            d_loss_real = discriminator.train_on_batch(batch_cells, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = gan.train_on_batch([batch_nuclei, batch_noise], np.ones((batch_size, 1)))

            print(f"Epoch {epoch}: [D loss: {d_loss[0]}, D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        if epoch % validation_interval == 0 and val_nuclei is not None:
            val_predictions = []
            for i in range(0, len(val_nuclei), batch_size):
                val_batch_nuclei = val_nuclei[i:i + batch_size]
                val_batch_noise = np.random.normal(0, 1, (batch_size, 256, 256, 1))
                val_generated_images = generator.predict([val_batch_nuclei, val_batch_noise])
                val_predictions.extend(val_generated_images)
            val_predictions = np.array(val_predictions)
            for j, img in enumerate(val_predictions):
                img = (img * 127.5 + 127.5).astype(np.uint8)
                output_image_path = f'improved_noise_training/predicted_image_{epoch}_{j}.png'
                cv2.imwrite(output_image_path, img)

        if epoch % save_interval == 0:
            generator.save(f'../models/GAN_combined_loss/improved_GAN_combined_loss_generator_epoch_{epoch}_g_loss_{g_loss}.h5')
            discriminator.save(f'../models/GAN_combined_loss/improved_GAN_combined_loss_discriminator_epoch_{epoch}_d_loss_{d_loss}.h5')

    generator.save('../models/combined_loss/improved_GAN_combined_loss_generator_final.h5')
    discriminator.save('../models/combined_loss/improved_GAN_combined_loss_discriminator_final.h5')

generator = GAN_combined_loss.build_generator()
discriminator = GAN_combined_loss.build_discriminator()
gan = GAN_combined_loss.build_gan(generator, discriminator)

discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss=GAN_combined_loss.combined_loss)

def load_images_from_directory(directory, size=(256, 256)):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
    return np.array(images)

train_nuclei = load_images_from_directory('../data/2_norm_images/nuclei/')
train_cells = load_images_from_directory('../data/2_norm_images/cell/')
val_nuclei = load_images_from_directory('../data/5_Val/nuclei/')
val_cells = load_images_from_directory('../data/5_Val/cell/')

train_gan(generator, discriminator, gan, epochs=100, batch_size=2, train_nuclei=train_nuclei, train_cells=train_cells, val_nuclei=val_nuclei, val_cells=val_cells)

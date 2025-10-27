import GAN_latent_space
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

def train_gan(generator, discriminator, gan, epochs, batch_size, train_nuclei, train_cells, val_nuclei=None, val_cells=None,
              validation_interval=10, save_interval=10):
    d_losses = []
    g_losses = []
    combined_g_losses = []
    for epoch in range(epochs):
        # Shuffle and batch data
        indices = np.arange(len(train_nuclei))
        np.random.shuffle(indices)
        train_nuclei = [train_nuclei[i] for i in indices]
        train_cells = [train_cells[i] for i in indices]

        for i in range(0, len(train_nuclei), batch_size):
            batch_nuclei = np.array(train_nuclei[i:i + batch_size])
            batch_cells = np.array(train_cells[i:i + batch_size])
            batch_latent = np.random.normal(0, 1, (batch_size, 100)) ## This is only a vector of 100x1 as a normal distribution

            # Generate fake images
            generated_images = generator.predict([batch_nuclei, batch_latent])
            print('generated_output:', generated_images.shape)

            
            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(batch_cells, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            

            # Train the generator
            g_loss = gan.train_on_batch([batch_nuclei, batch_latent], np.ones((batch_size, 1)))  
            
             # Generate fake images
          
            # Directly train the generator on the actual target cells using combined_loss
            combined_g_loss = generator.train_on_batch([batch_nuclei, batch_latent], batch_cells)
            
            # We want the discriminator to think the generated images are real


            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            combined_g_losses.append(combined_g_loss)
            # Print the progress
            print(f"Epoch {epoch}: [D loss: {d_loss[0]}, D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]   [combined g loss: {combined_g_loss}]")
            
            with open('../results/GAN_latent_space.txt', 'a') as f:
                f.write(f"Epoch {epoch + 1}, D Loss: {d_loss[0]:.4f}, D Accuracy: {100 * d_loss[1]:.4f}, "
                    f"G Loss: {g_loss:.4f}, combined_g_loss: {combined_g_loss:.4f}\n")
                f.close()

        # Optionally validate the model
        if epoch % validation_interval == 0 and val_nuclei is not None:
            val_predictions = []
            for i in range(0, len(val_nuclei), batch_size):
                val_batch_nuclei = np.array(val_nuclei[i:i + batch_size])
                val_batch_latent = np.random.normal(0, 1, (batch_size, 100))
                val_generated_images = generator.predict([val_batch_nuclei, val_batch_latent])
                val_predictions.append(val_generated_images)
                
            # Concatenate the list of arrays into a single array
            val_predictions = np.concatenate(val_predictions, axis=0)
            
        # Save the models at regular intervals
        if epoch % save_interval == 0:
            generator.save(f'../models/GAN_latent_space/improved_GAN_combined_loss_generator_epoch_{epoch}_g_loss_{g_loss}.h5')
            discriminator.save(f'../models/GAN_latent_space/improved_GAN_combined_loss_discriminator_epoch_{epoch}_d_loss_{d_loss}.h5')

    generator.save('../models/GAN_latent_space/improved_GAN_latent_space_generator_final.h5')
    discriminator.save('../models/GAN_latent_space/improved_GAN_latent_space_discriminator_final.h5')
    
    
 

# Compile models
generator = GAN_latent_space.build_generator(latent_dim=100)
discriminator = GAN_latent_space.build_discriminator()
gan = GAN_latent_space.build_gan(generator, discriminator)

generator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=GAN_latent_space.combined_loss)
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Load and preprocess training and validation data
def load_images_from_directory(directory, size=(256, 256)):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        if img is not None:
            img = cv2.resize(img, size)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
            images.append(img)
    return np.array(images)

train_nuclei = load_images_from_directory('../data/2_norm_images/nuclei/')
train_cells = load_images_from_directory('../data/2_norm_images/cell/')
val_nuclei = load_images_from_directory('../data/5_Val/nuclei/')
val_cells = load_images_from_directory('../data/5_Val/cell/')

# Train the GAN
train_gan(generator, discriminator, gan, epochs=100, batch_size=2, train_nuclei=train_nuclei, train_cells=train_cells, 
          val_nuclei=val_nuclei, val_cells=val_cells)

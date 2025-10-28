import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import Diffusion

def train_diffusion_model(model, epochs, batch_size, train_nuclei, train_cells, train_masks, beta_schedule, val_nuclei=None,
                          val_cells=None, val_masks=None, validation_interval=10, save_interval=10):
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    for epoch in range(epochs):
        for i in range(0, len(train_nuclei), batch_size):
            batch_nuclei = np.array(train_nuclei[i:i + batch_size])
            batch_cells = np.array(train_cells[i:i + batch_size])
            batch_masks = np.array(train_masks[i:i + batch_size])
            t = np.random.uniform(0, 1, size=(batch_size, 1, 1, 1)).astype(np.float32)  # Ensure t is float32
            noise = tf.random.normal(shape=batch_cells.shape, dtype=tf.float32)  # Ensure noise is float32
            x_noisy = Diffusion.forward_diffusion(batch_cells, t, beta_schedule)
            with tf.GradientTape() as tape:
                noise_pred = model([batch_nuclei, batch_masks])
                loss = Diffusion.diffusion_loss(noise, noise_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(f"Epoch {epoch}: [Loss: {loss.numpy()}]")
            
            
        if epoch % validation_interval == 0 and val_nuclei is not None:
            val_loss = 0
            for i in range(0, len(val_nuclei), batch_size):
                val_batch_nuclei = np.array(val_nuclei[i:i + batch_size])
                val_batch_cells = np.array(val_cells[i:i + batch_size])
                val_batch_masks = np.array(val_masks[i:i + batch_size])
                t = np.random.uniform(0, 1, size=(batch_size, 1, 1, 1)).astype(np.float32)  # Ensure t is float32
                noise = tf.random.normal(shape=val_batch_cells.shape, dtype=tf.float32)  # Ensure noise is float32
                x_noisy = Diffusion.forward_diffusion(val_batch_cells, t, beta_schedule)
                noise_pred = model([val_batch_nuclei, val_batch_masks])
                val_loss += Diffusion.diffusion_loss(noise, noise_pred).numpy()
            val_loss /= len(val_nuclei) // batch_size
            print(f"Validation Loss: {val_loss}")
            
            with open('../results/Diffusion.txt', 'a') as f:
                f.write(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Val_Loss: {val_loss:.4f},")
                f.close()
        if epoch % save_interval == 0:
            model.save(f'../models/Diffusion_model/diffusion_model_epoch_{epoch}.h5')
    model.save('../models/Diffusion_model//diffusion_model_final.h5')





def load_images_from_directory(directory, size=(1104, 1104)):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is not None:
            img = cv2.resize(img, size)
            img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
    return np.array(images)


def extract_patches(images, patch_size=(256, 256)):
    patches = []
    img_h, img_w = images.shape[1:3]
    for img in images:
        for i in range(0, img_h, patch_size[0]):
            for j in range(0, img_w, patch_size[1]):
                patch = img[i:i + patch_size[0], j:j + patch_size[1]]
                if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                    patches.append(patch)
    return np.array(patches)


train_nuclei = load_images_from_directory('../data/2_norm_images/nuclei/')
train_cells = load_images_from_directory('../data/2_norm_images/cell/')
train_masks = load_images_from_directory('../data/1_raw_annotations/cell/')
val_nuclei = load_images_from_directory('../data/5_Val/nuclei/')
val_cells = load_images_from_directory('../data/5_Val/cell/')
val_masks = load_images_from_directory('../data/6_val_annot/cell/')


train_nuclei_patches = extract_patches(train_nuclei)
train_cells_patches = extract_patches(train_cells)
train_masks_patches = extract_patches(train_masks)
val_nuclei_patches = extract_patches(val_nuclei)
val_cells_patches = extract_patches(val_cells)
val_masks_patches = extract_patches(val_masks)

encoder = Diffusion.build_encoder()
decoder = Diffusion.build_decoder()
diffusion_model = Diffusion.build_diffusion_model(encoder, decoder)

diffusion_model.compile()
beta_schedule = lambda t: Diffusion.noise_schedule(t)


diffusion_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=Diffusion.diffusion_loss)

train_diffusion_model(diffusion_model, epochs=100, batch_size=2, train_nuclei=train_nuclei_patches, train_cells=train_cells_patches, train_masks=train_masks_patches, beta_schedule=beta_schedule, val_nuclei=val_nuclei_patches, val_cells=val_cells_patches, val_masks=val_masks_patches)


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def build_encoder():
    input_layer = Input(shape=(256, 256, 1))  # Updated to 1 channel
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
   
    return model

def build_decoder():
    latent_input = Input(shape=(32, 32, 512))
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(latent_input)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', activation='tanh')(x)  # Updated to 1 channel
    model = Model(inputs=latent_input, outputs=x)
    
    return model

def noise_schedule(t, beta_start=0.1, beta_end=0.2):
    return beta_start + t * (beta_end - beta_start)

def forward_diffusion(x0, t, beta_schedule):
    beta_t = noise_schedule(t)
    noise = tf.random.normal(shape=tf.shape(x0), dtype=tf.float32)  # Ensure noise is float32
    x0 = tf.cast(x0, tf.float32)  # Ensure x0 is float32
    beta_t = tf.cast(beta_t, tf.float32)  # Ensure beta_t is float32
    return tf.sqrt(1.0 - beta_t) * x0 + tf.sqrt(beta_t) * noise

def build_diffusion_model(encoder, decoder):
    nuclei_input = Input(shape=(256, 256, 1))  # Updated to 1 channel
    mask_input = Input(shape=(256, 256, 1))    # Updated to 1 channel
    encoded = encoder(nuclei_input)  # Only nuclei_input is passed to the encoder
    decoded = decoder(encoded)
    return Model(inputs=[nuclei_input, mask_input], outputs=decoded)

def diffusion_loss(noise_true, noise_pred):
    return tf.reduce_mean(tf.square(noise_true - noise_pred))
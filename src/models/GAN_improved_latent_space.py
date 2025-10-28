import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import os

# Improved Generator with Residual Blocks
def residual_block(x, filters):
    res = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    return x

def build_generator():
    nuclei_input = Input(shape=(256, 256, 1))
    latent_input = Input(shape=(256, 256, 1))

    x = concatenate([nuclei_input, latent_input], axis=-1)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    filters=64
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    filters=128
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(1, (3, 3), padding='same', activation='tanh')(x)
    
    model = Model(inputs=[nuclei_input, latent_input], outputs=x)
    return model

# Improved Discriminator
def build_discriminator():
    input_layer = Input(shape=(256, 256, 1))
    
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model

# GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    nuclei_input = Input(shape=(256, 256, 1))
    latent_input = Input(shape=(256, 256, 1))
    generated_image = generator([nuclei_input, latent_input])
    gan_output = discriminator(generated_image)
    model = Model(inputs=[nuclei_input, latent_input], outputs=gan_output)
    return model

# Loss functions
def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def total_variation_loss(x):
    return tf.reduce_mean(tf.image.total_variation(x))

def combined_loss(y_true, y_pred):
    return perceptual_loss(y_true, y_pred) + 1e-6 * total_variation_loss(y_pred)

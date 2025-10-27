import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os

def build_generator():
    nuclei_input = Input(shape=(256, 256, 1))
    noise_input = Input(shape=(256, 256, 1))

    x = concatenate([nuclei_input, noise_input], axis=-1)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    filters = 64
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    filters = 128
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
    
    model = Model(inputs=[nuclei_input, noise_input], outputs=x)
    return model

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

def build_gan(generator, discriminator):
    discriminator.trainable = False
    nuclei_input = Input(shape=(256, 256, 1))
    noise_input = Input(shape=(256, 256, 1))
    generated_image = generator([nuclei_input, noise_input])
    gan_output = discriminator(generated_image)
    model = Model(inputs=[nuclei_input, noise_input], outputs=gan_output)
    return model

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def combined_loss(y_true, y_pred):
    return perceptual_loss(y_true, y_pred)
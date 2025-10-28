# Define and compile the model
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import os


def unet_model(output_channels):
    inputs = layers.Input(shape=[256, 256, 1])
    
    # Encoder
    down_stack = [
        layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(512, 3, strides=2, padding='same', activation='relu'),
    ]

    # Decoder
    up_stack = [
        layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
    ]

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model

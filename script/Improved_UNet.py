import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import numpy as np
import os
import numpy as np
from PIL import Image
import os


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    # Encoder
    down_stack = [
        tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1024, 3, strides=2, padding='same', activation='relu'),
    ]

    # Decoder
    up_stack = [
        tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
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
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse + ssim_loss

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
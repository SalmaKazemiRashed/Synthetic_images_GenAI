import Preprocessing
import Improved_UNet 
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import numpy as np
import os



# Custom callback to save loss and accuracy
class SaveMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('../results/UNet_LR_metrics_gray.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
                    f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}\n")

# Example usage
train_nuclei_dir = '../data/2_norm_images/nuclei/'
train_cell_dir = '../data/2_norm_images/cell/'
val_nuclei_dir = '../data/5_Val/nuclei/'
val_cell_dir = '../data/5_Val/cell/'

with tf.device('cpu:0'):
    train_dataset = Preprocessing.load_and_augment_dataset(train_nuclei_dir, train_cell_dir)
    val_dataset = Preprocessing.load_and_augment_dataset(val_nuclei_dir, val_cell_dir)

batch_size = 1
train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Define and compile the model
generator = Improved_UNet.unet_model(output_channels=1)
generator.compile(optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  loss=Improved_UNet.custom_loss,
                  metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_path = "../models/UNet_LR/UNet_LR_gray_{epoch:04d}_{val_loss:.4f}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

# Train the model with the checkpoint callback
save_metrics_callback = SaveMetrics()
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(Improved_UNet.scheduler)

generator.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[checkpoint_callback, save_metrics_callback, learning_rate_callback])
generator.save('../models/UNet_LR/UNet_LR_gray_final_generator.h5')
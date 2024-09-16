import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import os
from skimage import img_as_float32, img_as_ubyte

    
def normalize(image):
    # Normalize image
    percentile = 99.9
    high = np.percentile(image,percentile)
    low = np.percentile(image,100-percentile)
    
    img = np.minimum(high, image)
    img = np.maximum(low, img)
    img = (img-low)/(high-low)
    img = img_as_ubyte(img)
    return img

### Image augmentation
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def load_image(file_path, target_size=(1104, 1104)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  # Decode with a single channel
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255  # Normalize to [0, 1]
    return image

def create_patches(image, patch_size=256):
    patches = []
    img_height, img_width, _ = image.shape
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size, 1):
                patches.append(patch)
    return np.array(patches)

def reconstruct_image_from_patches(patches, img_shape, patch_size=256):
    padded_height = ((img_shape[0] + patch_size - 1) // patch_size) * patch_size
    padded_width = ((img_shape[1] + patch_size - 1) // patch_size) * patch_size
    
    reconstructed_image = np.zeros((padded_height, padded_width, 1), dtype=np.float32)
    patch_count = np.zeros((padded_height, padded_width, 1), dtype=np.float32)

    patch_idx = 0
    for i in range(0, padded_height, patch_size):
        for j in range(0, padded_width, patch_size):
            if patch_idx < len(patches):
                y_start = i
                x_start = j
                reconstructed_image[y_start:y_start + patch_size, x_start:x_start + patch_size] += patches[patch_idx]
                patch_count[y_start:y_start + patch_size, x_start:x_start + patch_size] += 1
                patch_idx += 1

    reconstructed_image /= np.maximum(patch_count, 1)
    reconstructed_image = reconstructed_image[:img_shape[0], :img_shape[1]]  # Crop to original size
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    return reconstructed_image

def data_generator(nuclei_dir, cell_dir, batch_size=1, patch_size=256):
    nuclei_filenames = sorted(os.listdir(nuclei_dir))
    cell_filenames = sorted(os.listdir(cell_dir))

    while True:
        for start in range(0, len(nuclei_filenames), batch_size):
            nuclei_batch = nuclei_filenames[start:start + batch_size]
            cell_batch = cell_filenames[start:start + batch_size]

            nuclei_patches = []
            cell_patches = []

            for nuclei_filename, cell_filename in zip(nuclei_batch, cell_batch):
                nuclei_image = load_image(os.path.join(nuclei_dir, nuclei_filename))
                cell_image = load_image(os.path.join(cell_dir, cell_filename))

                nuclei_image_patches = create_patches(nuclei_image.numpy())
                cell_image_patches = create_patches(cell_image.numpy())

                nuclei_patches.extend(nuclei_image_patches)
                cell_patches.extend(cell_image_patches)

            nuclei_patches = np.array(nuclei_patches)
            cell_patches = np.array(cell_patches)

            yield nuclei_patches, cell_patches

def load_and_augment_dataset(nuclei_dir, cell_dir, patch_size=256):
    def load_image_and_augment(file_path):
        image = load_image(file_path)
        patches = create_patches(image.numpy())
        return patches

    def process_patch(nuclei_patch, cell_patch):
        nuclei_patch = augment_image(nuclei_patch)
        cell_patch = augment_image(cell_patch)
        return nuclei_patch, cell_patch

    nuclei_patches = []
    cell_patches = []

    for nuclei_filename, cell_filename in zip(sorted(os.listdir(nuclei_dir)), sorted(os.listdir(cell_dir))):
        nuclei_image_patches = load_image_and_augment(os.path.join(nuclei_dir, nuclei_filename))
        cell_image_patches = load_image_and_augment(os.path.join(cell_dir, cell_filename))
        nuclei_patches.extend(nuclei_image_patches)
        cell_patches.extend(cell_image_patches)

    nuclei_patches = np.array(nuclei_patches)
    cell_patches = np.array(cell_patches)

    nuclei_dataset = tf.data.Dataset.from_tensor_slices(nuclei_patches)
    cell_dataset = tf.data.Dataset.from_tensor_slices(cell_patches)
    dataset = tf.data.Dataset.zip((nuclei_dataset, cell_dataset))

    dataset = dataset.map(process_patch)

    return dataset

def load_dataset(nuclei_dir, cell_dir, patch_size=256):
    def load_image_and_create_patches(file_path):
        image = load_image(file_path)
        patches = create_patches(image.numpy())
        return patches

    nuclei_patches = []
    cell_patches = []

    for nuclei_filename, cell_filename in zip(sorted(os.listdir(nuclei_dir)), sorted(os.listdir(cell_dir))):
        nuclei_image_patches = load_image_and_create_patches(os.path.join(nuclei_dir, nuclei_filename))
        cell_image_patches = load_image_and_create_patches(os.path.join(cell_dir, cell_filename))
        nuclei_patches.extend(nuclei_image_patches)
        cell_patches.extend(cell_image_patches)

    nuclei_patches = np.array(nuclei_patches)
    cell_patches = np.array(cell_patches)

    nuclei_dataset = tf.data.Dataset.from_tensor_slices(nuclei_patches)
    cell_dataset = tf.data.Dataset.from_tensor_slices(cell_patches)
    dataset = tf.data.Dataset.zip((nuclei_dataset, cell_dataset))

    return dataset





def load_and_augment_dataset(nuclei_dir, cell_dir, patch_size=256):
    def load_image_and_create_patches(file_path):
        image = load_image(file_path)
        patches = create_patches(image)
        return patches

    nuclei_patches = []
    cell_patches = []

    for nuclei_filename, cell_filename in zip(sorted(os.listdir(nuclei_dir)), sorted(os.listdir(cell_dir))):
        nuclei_image_patches = load_image_and_create_patches(os.path.join(nuclei_dir, nuclei_filename))
        cell_image_patches = load_image_and_create_patches(os.path.join(cell_dir, cell_filename))
        nuclei_patches.extend(nuclei_image_patches)
        cell_patches.extend(cell_image_patches)

    nuclei_patches = np.array(nuclei_patches)
    cell_patches = np.array(cell_patches)

    def augment_patches(nuclei_patch, cell_patch):
        nuclei_patch = augment_image(nuclei_patch)
        cell_patch = augment_image(cell_patch)
        return nuclei_patch, cell_patch

    nuclei_dataset = tf.data.Dataset.from_tensor_slices(nuclei_patches)
    cell_dataset = tf.data.Dataset.from_tensor_slices(cell_patches)
    dataset = tf.data.Dataset.zip((nuclei_dataset, cell_dataset))

    # Apply augmentation
    dataset = dataset.map(augment_patches, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
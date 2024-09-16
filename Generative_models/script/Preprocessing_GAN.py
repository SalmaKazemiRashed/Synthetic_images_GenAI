import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Data Preparation Functions ---

def extract_patches(image, patch_size=256):
    patches = []
    img_height, img_width, _ = image.shape
    for i in range(0, img_height - patch_size + 1, patch_size):
        for j in range(0, img_width - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

def prepare_data(nuclei_folder, cell_folder, patch_size=256):
    nuclei_images = load_images_from_folder(nuclei_folder)
    cell_images = load_images_from_folder(cell_folder)
    
    nuclei_patches = []
    cell_patches = []
    
    for nuclei_img, cell_img in zip(nuclei_images, cell_images):
        nuclei_patches.extend(extract_patches(nuclei_img, patch_size))
        cell_patches.extend(extract_patches(cell_img, patch_size))
    
    nuclei_patches = np.array(nuclei_patches)
    cell_patches = np.array(cell_patches)
    
    # Normalization
    nuclei_patches = (nuclei_patches - 127.5) / 127.5
    cell_patches = (cell_patches - 127.5) / 127.5
    
    return nuclei_patches, cell_patches

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = np.expand_dims(img, axis=-1)  # Add a channel dimension
            images.append(img)
    return images

def split_data(nuclei_patches, cell_patches, test_size=0.2):
    return train_test_split(nuclei_patches, cell_patches, test_size=test_size)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import UNet
from tensorflow.keras.optimizers import Adam

#

# Load the saved model
generator = UNet.unet_model(output_channels=1)

# Load the weights from the checkpoint file
model_path = "../models/UNet/Unet_gray_final_generator.h5"  # Example path to the checkpoint file
generator.load_weights(model_path)

# Function to load and preprocess an image
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [1104, 1104])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def extract_patches(image, patch_size=256, step=256):
    patches = []
    positions = []
    img_height, img_width, _ = image.shape

    for i in range(0, img_height, step):
        for j in range(0, img_width, step):
            patch = np.zeros((patch_size, patch_size, 1), dtype=np.float32)
            patch[:min(patch_size, img_height - i), :min(patch_size, img_width - j)] = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches), positions

def reassemble_patches(patches, positions, img_shape, patch_size=256, step=256):
    reconstructed_image = np.zeros(img_shape, dtype=np.float32)
    patch_count = np.zeros(img_shape, dtype=np.float32)
    
    for patch, (i, j) in zip(patches, positions):
        reconstructed_image[i:i+patch_size, j:j+patch_size] += patch[:img_shape[0]-i, :img_shape[1]-j]
        patch_count[i:i+patch_size, j:j+patch_size] += 1
    
    reconstructed_image /= np.maximum(patch_count, 1)
    return reconstructed_image

def predict_full_image(generator, input_image, patch_size=256, step=256):
    input_patches, positions = extract_patches(input_image, patch_size, step)
    generated_patches = generator.predict(input_patches)
    
    output_image = reassemble_patches(generated_patches, positions, input_image.shape, patch_size, step)
    return output_image

# Load validation images
val_nuclei_dir = glob.glob('../data/5_Val/nuclei/*.png')
val_cell_dir = glob.glob('../data/5_Val/cell/*.png')

print('Herereeeeeeeee')
for img in val_nuclei_dir:
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(img), cmap='gray')
    plt.axis('off')
    plt.title('Original Nucleus')

    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = input_image / 255.0  # Normalize to [0, 1]

    input_image = np.expand_dims(input_image, axis=-1)
    
    predicted_image = predict_full_image(generator, input_image)
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Generated Cells')

    output_image_path = '../results/UNet/' + os.path.basename(img)
    cv2.imwrite(output_image_path, (predicted_image * 255).astype(np.uint8))
    
    plt.subplot(1, 3, 3)
    ground_truth_img = cv2.imread('../script/data/5_Val/cell/' + os.path.basename(img).replace('d0', 'd1'), cv2.IMREAD_GRAYSCALE)
    plt.imshow(ground_truth_img, cmap='gray')
    plt.axis('off')
    plt.title('Original Cells')
    plt.savefig('../results/UNet/'+os.path.basename(img).replace('.png', '_prediction.png'))
    
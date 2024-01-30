import random

import cv2

import csv
import os

import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import os

# Initialize an empty list to store the data dictionaries
data = []

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices) > 0:
    print("gpu")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Open the CSV file
csv_file_path = './mydata/image_mask_paths.csv'  # Replace with your CSV file path
with open(csv_file_path, 'r') as csvfile:
    csv_reader = list(csv.DictReader(csvfile))  # Read the CSV into a list of dictionaries

    for row in csv_reader:


        # Assuming "images" and "masks" columns contain file paths
        Ap = row['images']
        maskA = row['masks']
        A = row['images'].replace("mydata", "mydata2")

        print(Ap.split("/images")[0])
        # Find rows with the same prefix in 'images' column
        matching_rows=[]
        for r in csv_reader:

            matching_rows.append(r)
        print(matching_rows)
        # Loop through matching rows and append data
        for matching_row in matching_rows:


            Bp =  matching_row['images']
            maskB =  matching_row['masks']
            B = matching_row['images'].replace("mydata", "mydata2")

            data.append({'mask_path': B,
                         'realmask_path': maskB,
                         'image_path': Bp,
                         'realmask_path2': maskB,
                         'mask_path2': A,
                         'realmask_path3': maskA ,
                         'image_path2': Ap,
                         'realmask_path4': maskA })



# Define image dimensions
img_width, img_height = 128, 128
channels = 3

print("combined")
print(data.__len__())


random_indices = random.sample(range(len(data)), 5)

# Plot the images
fig, axes = plt.subplots(5, 4, figsize=(15, 15))

for i, index in enumerate(random_indices):
    row = data[index]

    # Plot the first image
    axes[i, 0].imshow(plt.imread(row['image_path']))
    axes[i, 0].set_title(f'Image {i + 1}')
    axes[i, 0].axis('off')

    # Plot the corresponding mask
    axes[i, 1].imshow(plt.imread(row['mask_path']))
    axes[i, 1].set_title(f'Mask {i + 1}')
    axes[i, 1].axis('off')

    # Plot the second image
    axes[i, 2].imshow(plt.imread(row['image_path2']))
    axes[i, 2].set_title(f'Image2 {i + 1}')
    axes[i, 2].axis('off')

    # Plot the corresponding second mask
    axes[i, 3].imshow(plt.imread(row['mask_path2']))
    axes[i, 3].set_title(f'Mask2 {i + 1}')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()
# Initialize empty lists to store images and masks
images1 = []
result = []
masks2 = []
masks = []


def preprocess_image(image_path, mask_path, img_width, img_height):
    # Load the image and mask
    img = load_img(image_path, target_size=(img_width, img_height))
    mask = load_img(mask_path, target_size=(img_width, img_height), color_mode='grayscale')  # Assuming the mask is grayscale

    # Convert to arrays
    img_array = img_to_array(img)
    mask_array = img_to_array(mask)

    # Apply the mask and crop the image
    img_array_cropped = apply_and_crop_mask(img_array, mask_array)

    # Expand dimensions, normalize, and return
    img_array_cropped = np.expand_dims(img_array_cropped, axis=0)
    img_array_cropped = img_array_cropped.astype('float32') / 255.0

    return img_array_cropped

def apply_and_crop_mask(img_array, mask_array):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_array[:, :, 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the contours
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image based on the bounding box
        img_array_cropped = img_array[y:y+h, x:x+w]

        # Resize the cropped image to the original size (img_width, img_height)
        img_array_cropped = cv2.resize(img_array_cropped, (img_array.shape[1], img_array.shape[0]))

        return img_array_cropped

    return img_array  # Return the original image if no contours found


for item in data:
    image1_path = item['image_path']
    image2_path = item['image_path2']
    mask2_path = item['mask_path2']
    mask_path = item['mask_path']
    realmask1 = item['realmask_path']
    realmask2 = item['realmask_path2']
    realmask3 = item['realmask_path3']
    realmask4 = item['realmask_path4']
    # Load and resize images using load_img
    image1 = preprocess_image(image1_path,realmask2, img_width, img_height)
    image2 = preprocess_image(image2_path,realmask4, img_width, img_height)

    mask = preprocess_image(mask_path,realmask1, img_width, img_height)

    mask2 = preprocess_image(mask2_path,realmask3, img_width, img_height)

    images1.append(image1)

    masks.append(mask)
    masks2.append(mask2)
    result.append(image2)

print("loaded")
images1 = np.array(images1)
result = np.array(result)
masks2 = np.array(masks2)
masks = np.array(masks)

# Load the saved models
encoder = load_model('encoder_model.h5')
decoder = load_model('decoder_model.h5')
modified_encodings = []
for i in range(len(masks)):
    # Assuming you want to calculate modified encoding for each pair of images and masks
    image_a = masks[i]  # Image from images1
    image_b = images1[i]  # Image from result

    # Encode images
    encoded_img_a = encoder.predict(image_a)
    encoded_img_b = encoder.predict(image_b)

    # Calculate modified encoding
    modified_encoding = encoded_img_b - encoded_img_a

    # Append modified encoding to the list
    modified_encodings.append(modified_encoding)

modified_encodings = np.array(modified_encodings)
print("modified_encodings")
encoded_masks2 = []
encoded_results = []

for i in range(len(modified_encodings)):
    # Assuming you want to calculate the encoding for each pair of modified encoding and masks2/result images
    modified_encoding = modified_encodings[i]
    mask2_image = masks2[i]
    result_image = result[i]

    # Encode masks2 image
    encoded_mask2 = encoder.predict(mask2_image)

    # Encode result image
    encoded_result = encoder.predict(result_image)

    # Append encoded masks2 and result to the respective lists
    encoded_masks2.append(encoded_mask2)
    encoded_results.append(encoded_result)

encoded_masks2 = np.array(encoded_masks2)
encoded_results = np.array(encoded_results)

print("encoded_results")


print("Shape of encoded_masks2:", encoded_masks2.shape)
print("Shape of modified_encodings:", modified_encodings.shape)



encoded_masks2_shape = encoded_masks2.shape[1:]
modified_encodings_shape = modified_encodings.shape[1:]

# Define input layers
input_encoded_masks2 = keras.Input(shape=encoded_masks2_shape)
input_modified_encodings = keras.Input(shape=modified_encodings_shape)

# Combine the input layers
concatenated_inputs = keras.layers.Concatenate()([input_encoded_masks2, input_modified_encodings])

# Check the shape after concatenation
print("Shape of concatenated_inputs:", concatenated_inputs.shape)

# If there is an extra dimension, squeeze it out
concatenated_inputs = tf.squeeze(concatenated_inputs, axis=[1])  # Adjust the axis if necessary

# Input layer
input_layer = concatenated_inputs

# Convolutional layers with increasing depth
conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
conv1 = layers.BatchNormalization()(conv1)
conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
conv2 = layers.BatchNormalization()(conv2)
conv2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
conv3 = layers.BatchNormalization()(conv3)
conv3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv3)
conv4 = layers.BatchNormalization()(conv4)
conv4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
conv5 = layers.BatchNormalization()(conv5)
conv5 = layers.MaxPooling2D(pool_size=(2, 2))(conv5)

# Flatten the output of the last convolutional layer
flatten_layer = layers.Flatten()(conv5)

# Dense (Fully Connected) layers
dense1 = layers.Dense(256, activation='relu')(flatten_layer)
dense1 = layers.Dropout(0.5)(dense1)

dense2 = layers.Dense(256, activation='relu')(dense1)
dense2 = layers.Dropout(0.5)(dense2)

dense3 = layers.Dense(128, activation='relu')(dense2)

# Output layer
output_layer = layers.Dense(np.prod(encoded_results.shape[1:]), activation='relu')(dense3)
output_layer = layers.Reshape(encoded_results.shape[1:])(output_layer)



# Create the model
model = keras.models.Model(inputs=[input_encoded_masks2, input_modified_encodings], outputs=output_layer)



model_output_shape = model.output_shape
target_data_shape = encoded_results.shape
print("Model output shape:", model_output_shape)
print("Target data shape:", target_data_shape)

# Compile the model
# Compile the model with accuracy metric
# If your task is regression, you might want to use a different metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Define the target encoded_results
target_encoded_results = encoded_results


print("train")
# Train the model
history = model.fit([encoded_masks2, modified_encodings], target_encoded_results, epochs=250, batch_size=32,
                    validation_split=0.2)


# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.title('Training and Validation Loss/Accuracy')
plt.show()


num_samples = encoded_results.shape[0]


sample_indices = np.random.choice(num_samples, 10, replace=False)

predicted_encoded_results = model.predict([encoded_masks2[sample_indices], modified_encodings[sample_indices]])

for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(18, 12))  # Increase figure size for a 3x2 layout

    # Decode the original and predicted encoded results
    encoded_results_dec = decoder.predict(encoded_results[idx])
    predicted_encoded_results_dec = decoder.predict(predicted_encoded_results[i])

    # Remove the batch dimension (1) from the image data
    encoded_results_dec_squeezed = np.squeeze(encoded_results_dec, axis=0)
    predicted_encoded_results_dec_squeezed = np.squeeze(predicted_encoded_results_dec, axis=0)

    # Original encoded results
    plt.subplot(3, 2, 1)
    plt.imshow(encoded_results_dec_squeezed)
    plt.title(f'Original Encoded Result {idx}')

    # Predicted encoded results
    plt.subplot(3, 2, 2)
    plt.imshow(predicted_encoded_results_dec_squeezed)
    plt.title(f'Predicted Encoded Result {idx}')

    # Display mask
    plt.subplot(3, 2, 3)
    plt.imshow(masks2[idx].squeeze())
    plt.title(f'Encoded Masks2 {idx}')

    # Display input image
    plt.subplot(3, 2, 4)
    plt.imshow(images1[idx].squeeze())
    plt.title(f'Input Image {idx}')

    # Display mask
    plt.subplot(3, 2, 5)
    plt.imshow(masks[idx].squeeze())
    plt.title(f'Encoded Masks1 {idx}')

    # Display input image
    plt.subplot(3, 2, 6)
    plt.imshow(result[idx].squeeze())
    plt.title(f'Input Image2 {idx}')

    plt.show()


deep_model = 'deep_model.h5'

# Save the encoder and decoder models
model.save(deep_model)
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

# Ensure that GPU is being used if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    print("gpu")
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Set the path to your image folder
img_folder = "data2"  # Replace with your folder path

# Define image dimensions
img_width, img_height = 128, 128
channels = 3

# Load and preprocess images from the folder
def load_images_from_folder(folder):

    images = []
    for filename in os.listdir(folder):
        try:
         img = load_img(os.path.join(folder, filename), target_size=(img_width, img_height))
         img_array = img_to_array(img)
         images.append(img_array)
        except :
            print("33")
    return np.array(images)

# Load images from the folder
images = load_images_from_folder(img_folder)

# Normalize images
images = images.astype('float32') / 255.0

# Split the dataset into training and validation sets
X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)

# Building the Autoencoder with GPU
with tf.device('/GPU:0'):
    # Encoder
    input_img = Input(shape=(img_width, img_height, channels))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoder = Model(input_img, encoded)

    # Decoder
    decoder_input = Input(shape=(int(img_width / 2), int(img_height / 2), 64))  # Adjust the shape according to the encoder's output
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, decoded)

    # Autoencoder
    autoencoder_input = Input(shape=(img_width, img_height, channels))
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = Model(autoencoder_input, decoded_img)

    # Compile the model
    autoencoder.compile(optimizer=Adam(lr=0.001), loss='mse')

# Train the model
epochs = 50
batch_size = 16
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val))

# Encode and decode images
decoded_imgs = autoencoder.predict(X_val)

# Plot the original and reconstructed images
n = 5  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val[i])
    plt.title('Original')
    plt.axis('off')

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title('Reconstructed')
    plt.axis('off')

plt.show()

# Define the paths for saving models
encoder_path = 'encoder_model.h5'
decoder_path = 'decoder_model.h5'

# Save the encoder and decoder models
encoder.save(encoder_path)
decoder.save(decoder_path)

print("Encoder and Decoder models saved successfully.")

import numpy as np
import pickle
import cv2
import os
from glob import glob
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Clear previous TensorFlow sessions and set logging level
K.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    """
    Get the size of the images from a sample image.
    """
    img = cv2.imread('gestures/1/100.jpg', cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img.shape
    else:
        raise FileNotFoundError("Sample image not found. Please check the file path.")

def get_num_of_classes():
    """
    Get the number of classes by counting subdirectories in 'gestures/'.
    """
    return len(glob('gestures/*'))

image_x, image_y = get_image_size()

def cnn_model(num_of_classes):
    """
    Create and compile the CNN model.
    """
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Define ModelCheckpoint callback
    filepath = "cnn_model_keras2.keras"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list

import numpy as np

def normalize_labels(labels, num_classes):
    # Normalize labels to fit in the range [0, num_classes - 1]
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_mapping[label] for label in labels])

def train():
    try:
        with open("train_images", "rb") as f:
            train_images = np.array(pickle.load(f))
        with open("train_labels", "rb") as f:
            train_labels = np.array(pickle.load(f), dtype=np.int32)
        with open("val_images", "rb") as f:
            val_images = np.array(pickle.load(f))
        with open("val_labels", "rb") as f:
            val_labels = np.array(pickle.load(f), dtype=np.int32)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    num_of_classes = get_num_of_classes()
    
    print(f"Number of classes: {num_of_classes}")
    print(f"Train labels: {np.unique(train_labels)}")
    print(f"Validation labels: {np.unique(val_labels)}")

    # Normalize labels to fit in the number of classes
    train_labels = normalize_labels(train_labels, num_of_classes)
    val_labels = normalize_labels(val_labels, num_of_classes)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    train_labels = to_categorical(train_labels, num_classes=num_of_classes)
    val_labels = to_categorical(val_labels, num_classes=num_of_classes)

    print(f'Validation labels shape: {val_labels.shape}')

    model, callbacks_list = cnn_model(num_of_classes)
    model.summary()
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=32, callbacks=callbacks_list)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    model.save('cnn_model_keras2.h5')

if __name__ == "__main__":
    train()

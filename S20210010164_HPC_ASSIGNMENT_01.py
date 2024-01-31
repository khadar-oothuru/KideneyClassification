import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import cProfile
import pstats

def load_images(image_directory, class_folder, label_value, dataset, label):
    images = os.listdir(os.path.join(image_directory, class_folder))
    for image_name in images:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, class_folder, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(label_value)

def preprocess_data(image_directory):
    dataset = []
    label = []

    # Load normal images
    load_images(image_directory, 'Normal', 0, dataset, label)

    # Load tumor images
    load_images(image_directory, 'Tumor', 1, dataset, label)

    # Load cyst images
    load_images(image_directory, 'Cyst', 2, dataset, label)

    # Load stone images
    load_images(image_directory, 'Stone', 3, dataset, label)

    dataset = np.array(dataset)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    return x_train, x_test, y_train, y_test

def build_model(input_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(input_size, input_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=2, batch_size=16):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              verbose=1, epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False)

    return model

def save_model(model, filename):
    model.save(filename)

if __name__ == "__main__":
    INPUT_SIZE = 64
    image_directory = 'datasets/'

    
    cProfile.run("x_train, x_test, y_train, y_test = preprocess_data(image_directory)", "preprocess_profile")

  
    stats_preprocess = pstats.Stats("preprocess_profile")
    stats_preprocess.strip_dirs().sort_stats("cumulative").print_stats(20)

    model = build_model(INPUT_SIZE)

   
    cProfile.run("trained_model = train_model(model, x_train, y_train, x_test, y_test)", "train_profile")

    
    stats_train = pstats.Stats("train_profile")
    stats_train.strip_dirs().sort_stats("cumulative").print_stats(20)

    save_model(trained_model, 'trainedModel.h5')

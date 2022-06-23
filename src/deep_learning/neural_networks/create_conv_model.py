from tensorflow.keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Flatten, GRU, Dense, Activation, Dropout, Reshape, Permute
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf


def create_conv_model(img_shape, all_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',input_shape=(*img_shape, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(*img_shape, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape = (*img_shape, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape = (*img_shape, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(all_classes), activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    return model

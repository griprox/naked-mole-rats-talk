from tracing_model.image_and_trace_processing import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import  UpSampling2D, Concatenate, Input
import os 

def predict_with_augmentation(images, model, times_augment = 100):
    
    """ Traces all images, augmenting each of them """
    
    result = []
    model_img_shape = model.input_shape[1 : -1]
    
    for im in images:
        preds = []
        im_copies = []
        shifts = []
        resizing = False
        shape_orig = im.shape
        
        if im.shape != model.input_shape[1 : -1]:
            im_rs = resize(im, model.input_shape[1 : -1])
            resizing = True
        else:
            im_rs = im
            resizing = False
            
        for _ in range(5):
            
            shift_vertical = np.random.randint(-int(im_rs.shape[0] / 36), int(im_rs.shape[0] / 36))
            shift_horizontal = np.random.randint(-int(im_rs.shape[1] / 12), int(im_rs.shape[1] / 12))
            
            im_aug = np.roll(np.roll(im_rs, shift_vertical, axis = 0), shift_horizontal, axis = 1)
            im_copies.append(im_aug)
            shifts.append((shift_vertical, shift_horizontal))
            
        traces = model.predict(np.reshape(im_copies,  (-1, *model_img_shape, 1)))[:, :, :, 0]
        traces = np.array([np.roll(np.roll(t, -s_v, axis = 0), -s_h, axis = 1) 
                  for (t, (s_v, s_h)) in zip(traces, shifts)])
        final_trace = np.product(traces + 0.5, 0)
        result.append(resize(final_trace, shape_orig))
    return result


def load_model(name = 'segmentator3000',
               path = '/home/nakedmoleratvoices/Mole rats reborn/CodeRefactoredFinal/tracing_model/'):
    
    """ Tries to load model"""
    
    if name in os.listdir(path):
        model = tf.keras.models.load_model(path + name)
        print('Succesfully loaded model')
    else:
        print('Model was not found')
    return model

def create_autoencoder(image_shape):
    
    input_img = Input(shape = (*image_shape, 1))

    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    encoded = MaxPooling2D((2, 2), padding = 'same')(x)

    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = None, padding = 'same')(x)

    model = tf.keras.Model(input_img, decoded)
    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(lr = 5e-3))
    model.summary()
    return model
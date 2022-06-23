from datetime import datetime
import pandas as pd
import os
import librosa
import numpy as np
from skimage.transform import resize

def rgb2gray(rgb):
    """ Transforms RGB image into grayscale """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_ordered_y(y, order):
    """ Orders y according to order """
    if order == None:
        return sorted(list(set(y)))
    else:
        return [yi for yi in order if yi in np.array(y)]
    
def get_balancing_indexes(y):
    """ Computes flat list of indexes, s.t. y[indexes] is balanced """
    bc = np.bincount(y)
    min_count = np.min(bc[bc > 0])
    indexes = []
    for yi in range(len(bc)):
        yi_inds = np.where(y == yi)[0]
        indexes.append(yi_inds[:min_count])
    return np.concatenate(indexes)

def get_grouping_indexes(grouping_column, shuffle = True):
    """ Computes list of lists of indexes, s.t. each each internal list contains all indexes
    for single value in the grouping column"""
    indexes = [np.where(grouping_column == val)[0] for val in set(grouping_column)]
    if shuffle:
        np.random.shuffle(indexes)
    return np.array(indexes)

def date_to_datetime(date_string):
    """ Transforms date string into datetime object """
    date_string_uni = date_string.replace('-', '.').replace('/', '.').replace('_', '.')
    date_list_of_strings = date_string_uni.split('.')
    date_list_of_ints = list(map(int, date_list_of_strings))
    date_list_of_ints[-1] = int(date_list_of_ints[-1] + 2000)
    return datetime(*reversed(date_list_of_ints))


def delete_indexes(indexes, *arrs, reset_dfs_indexes = True):
    """ Drops indexes from arrays/lists/dataframes arrs """
    result = []
    for ar in arrs:
        if isinstance(ar, pd.DataFrame):
            ar_new =  ar.drop(indexes, axis = 0)
            if reset_dfs_indexes:
                ar_new = ar_new.reset_index(drop = True)
        else:
            ar_new = [ar[ind] for ind in range(len(ar)) if ind not in indexes]
        result.append(ar_new)
    return result


def overwrite_recs_as_npy(path, sr=22050):
    """ Overwrites all recordings in path in .wav format with .npy """
    rec_names = os.listdir(path)
    rec_names_wav = [rn for rn in rec_names if '.wav' in rn]
    rec_names_npy = [rn for rn in rec_names if '.npy' in rn]
    print('Found %d .wav and %d .npy recordings. Wav ones will be overwritten with npy format.' % 
          (len(rec_names_wav), len(rec_names_npy)))
    wavs_to_overwrite = [rn for rn in rec_names_wav if rn.replace('wav', 'npy') not in rec_names_npy]
    for rn in wavs_to_overwrite:
        rec, _ = librosa.load(path + rn, sr = sr)
        np.save(path + rn[:-3] + 'npy', rec)
        os.remove(path + rn)
        
def make_image(s):
    """ Generates image from sound. This function should ONLY be used for visualizations, not for analysis"""
    D = np.abs(librosa.stft(s, n_fft=512))
    D = np.flip(D, 0)
    output_shape = (int(D.shape[0] / 2), int(D.shape[1] * 1.5))
    im = librosa.amplitude_to_db(D, ) 
    return resize(im, output_shape)
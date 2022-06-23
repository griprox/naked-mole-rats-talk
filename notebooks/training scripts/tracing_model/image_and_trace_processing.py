from skimage.transform import resize
import numpy as np


def binarize_trace(trace):
    trace_copy = np.array(trace)
    trace_copy[trace > 0] = 1
    return trace_copy

def augment_im_and_trace(im, trace):
    im_aug, trace_aug = roll_im_and_trace(im, trace)
    im_aug, trace_aug = zoom_im_and_trace(im_aug, trace_aug)
    im_aug = random_freq_mask(im_aug)
    im_aug = random_time_mask(im_aug)
    return im_aug, trace_aug


    
def roll_im_and_trace(im, trace):
    """ Randomly rolls both image and trace """
    
    max_to_bot = im.shape[0] - np.max(np.where(trace)[0])
    max_to_top = np.min(np.where(trace)[0])
    max_to_left = np.min(np.where(trace)[1])
    max_to_right = im.shape[1] - np.max(np.where(trace)[1])
    
    shift_vertical = np.random.randint(-max_to_top / 3, 3 * max_to_bot / 5)
    shift_horizontal = np.random.randint(-max_to_left, max_to_right)
    
    im_shifted = np.roll(np.roll(im, shift_vertical, axis = 0), shift_horizontal, axis = 1)
    trace_shifted = np.roll(np.roll(trace, shift_vertical, axis = 0), shift_horizontal, axis = 1)
    
    return im_shifted, trace_shifted

def roll_im(im):
    
    max_to_bot = im.shape[0] / 10
    max_to_top = im.shape[0] / 10
    max_to_left = im.shape[1] / 10
    max_to_right = im.shape[1] / 10
    
    shift_vertical = np.random.randint(-max_to_top, max_to_bot)
    shift_horizontal = np.random.randint(-max_to_left, max_to_right)
    
    im_shifted = np.roll(np.roll(im, shift_vertical, axis = 0), shift_horizontal, axis = 1)
    
    return im_shifted
    
    
def zoom_im(im):
    rate = np.random.uniform(0, 0.15)
    a, b = im.shape
    a_max = int(rate * a)
    b_max = int(rate * b)
    
    a_s = np.random.randint(0, a_max + 1)
    b_s = np.random.randint(0, b_max + 1)
    
    a_e = -1 - np.random.randint(0, a_max + 1)
    b_e = b - np.random.randint(0, b_max + 1)
    
    im_zoom = im[a_s : a_e, b_s : b_e]
    return im

    
def zoom_im_and_trace(im, trace):
    ''' Performs zoom with random rate '''
    rate = np.random.uniform(0, 0.15)
    a, b = im.shape
    a_max = int(rate * a)
    b_max = int(rate * b)
    
    a_s = np.random.randint(0, a_max + 1)
    b_s = np.random.randint(0, b_max + 1)
    
    a_e = -1 - np.random.randint(0, a_max + 1)
    b_e = b - np.random.randint(0, b_max + 1)
    
    im_zoom = im[a_s : a_e, b_s : b_e]
    trace_zoom = trace[a_s : a_e, b_s : b_e]
    
    return resize(im_zoom, im.shape), resize(trace_zoom, im.shape)

def random_time_mask(im):
    ''' Masks out random time band '''
    im_copy = np.copy(im)
    
    width = np.random.choice([0, 1, 2, 3], p = [0.5, 0.23, 0.23, 0.04])
    width = min(width, int(im.shape[1]//2))
    mask_start = np.random.randint(0, im.shape[1] - width)
    mask = np.zeros(im.shape, dtype = 'bool')
    mask[:, np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] =  np.mean(im_copy[mask])
    
    return im_copy

def random_freq_mask(im):
    ''' Masks out random frequency band '''
    im_copy = np.copy(im)
    
    width = np.random.choice([0, 1, 2, 3], p = [0.5, 0.23, 0.23, 0.04])
    mask_start = np.random.randint(0, im.shape[0] - width)
    mask = np.zeros(im.shape, dtype = 'bool')
    mask[np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] =  np.mean(im_copy[mask])
    
    return im_copy
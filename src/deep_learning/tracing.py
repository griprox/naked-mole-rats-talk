import numpy as np
from skimage.transform import resize

def threshold_le(trace, th):
    """Sets trace to be 0 at points where it is <= th"""
    new_trace = np.copy(trace)
    mask = trace <= th
    new_trace[mask] = 0
    new_trace[~mask] = 1
    return new_trace

def clear_trace(trace, th = 0.2):
    """ Tries to delete incorrectly traced parts leaving only the main harmonic"""
    trace = threshold_le(trace, th)
    for i in range(trace.shape[1]):
        inds = np.where(trace[:, i] > 0)[0]
        if len(inds) ==0:
            continue
        max_ind = max(inds)
        inds_bad = [ind for ind in inds if max_ind - ind > 10]
        trace[inds_bad, i] = 0
    return trace

def roll_pair_of_ims(im1, im2, shift0 = -1, shift1 = -1):
    """ Rolls iamges 1 and 2 by the same intervals in 2 dimensions """
    if shift1 == -1:
        N = im1.shape[1]
        shift_max = int(N / 6)
        shift = np.random.randint(-shift_max, shift_max)
       
    im1_s = np.roll(im1, shift, axis = 1)
    im2_s = np.roll(im2, shift, axis = 1)

    if shift0 == -1:
        N = im1_s.shape[0]
        shift_max = int(N / 8)
        shift = np.random.randint(-shift_max, shift_max)
    return np.roll(im1_s, shift, axis = 0), np.roll(im2_s, shift, axis = 0)

def predict_with_augmentation(images, model, times_augment = 5):
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
            
        for _ in range(times_augment):
            
            shift_vertical = np.random.randint(-int(im_rs.shape[0] / 36), int(im_rs.shape[0] / 36))
            shift_horizontal = np.random.randint(-int(im_rs.shape[1] / 12), int(im_rs.shape[1] / 12))
            
            im_aug = np.roll(np.roll(im_rs, shift_vertical, axis = 0), shift_horizontal, axis = 1)
            im_copies.append(im_aug)
            shifts.append((shift_vertical, shift_horizontal))
            
        traces = model.predict(np.reshape(im_copies,  (-1, *model_img_shape, 1)))[:, :, :, 0]
        traces = np.array([np.roll(np.roll(t, -s_v, axis = 0), -s_h, axis = 1) 
                  for (t, (s_v, s_h)) in zip(traces, shifts)])
        final_trace = np.mean(traces, 0)
        result.append(resize(final_trace, shape_orig))
    return result
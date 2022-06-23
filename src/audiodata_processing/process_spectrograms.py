import numpy as np
import librosa


def random_roll(im, dim1_factor=12, dim0_factor=16):
    """ Randomly rolls image in both dimensions """
    dim1_shift_max = int(im.shape[1] / dim1_factor)
    dim1_shift = np.random.randint(-dim1_shift_max, dim1_shift_max + 1)
    im_shifted = np.roll(im, dim1_shift, axis=1)

    dim0_shift_max = im_shifted.shape[0]
    dim0_shift_max = int(dim0_shift_max / dim0_factor)
    shift = np.random.randint(-dim0_shift_max, dim0_shift_max // 2 + 1)
    return np.roll(im_shifted, shift, axis=0)


def random_zoom(im, max_rate=0.15):
    """ Performs zoom with random rate """
    rate = np.random.uniform(0, max_rate)
    a, b = im.shape
    a_max = int(rate * a)
    b_max = int(rate * b)

    a_s = np.random.randint(0, a_max + 1)
    b_s = np.random.randint(0, b_max + 1)
    a_e = -1 - np.random.randint(0, a_max + 1)
    b_e = b - np.random.randint(0, b_max + 1)
    return im[a_s: a_e, b_s: b_e]


def random_time_mask(im, width_probs=((0, .5), (1, 0.23), (2, 0.23), (3, .04))):
    """ Masks out random time band """
    im_copy = np.copy(im)
    widths = [el[0] for el in width_probs]
    probs = [el[1] for el in width_probs]
    width = np.random.choice(widths, p=probs)
    width = min(width, int(im.shape[1] // 2))

    mask_start = np.random.randint(0, im.shape[1] - width)
    mask = np.zeros(im.shape, dtype='bool')
    mask[:, np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] = np.mean(im_copy[mask])
    return im_copy


def random_freq_mask(im, width_probs=((0, .5), (1, 0.23), (2, 0.23), (3, .04))):
    """ Masks out random frequency band """
    im_copy = np.copy(im)
    widths = [el[0] for el in width_probs]
    probs = [el[1] for el in width_probs]
    width = np.random.choice(widths, p=probs)

    mask_start = np.random.randint(0, im.shape[0] - width)
    mask = np.zeros(im.shape, dtype='bool')
    mask[np.arange(mask_start, mask_start + width)] = True
    im_copy[mask] = np.mean(im_copy[mask])
    return im_copy


def resize_with_padding(im, target_shape, val=0):
    """ Transform image to target_shape.
        To decrease a dimension this function crops image, to increase -- pads with fixed value 'val' """
    if target_shape is None:
        return im
    diff0 = target_shape[0] - im.shape[0]
    diff1 = target_shape[1] - im.shape[1]
    if diff0 == 0:
        im_rs = im

    elif diff0 < 0:
        start0 = np.random.randint(0, -diff0)
        end0 = -diff0 - start0
        im_rs = im[start0: -end0]
    else:
        pad_top = np.random.randint(0, diff0)
        pad_bot = diff0 - pad_top
        im_rs = np.concatenate([val * np.ones((pad_top, im.shape[1])), im, val * np.ones((pad_bot, im.shape[1]))])

    if diff1 == 0:
        pass

    elif diff1 < 0:
        start1 = np.random.randint(0, -diff1)
        end1 = -diff1 - start1
        im_rs = im_rs[:, start1: -end1]
    else:
        pad_left = np.random.randint(0, diff1)
        pad_right = diff1 - pad_left
        im_rs = np.concatenate([val * np.ones((im_rs.shape[0], pad_left)),
                                im_rs, val * np.ones((im_rs.shape[0], pad_right))], 1)
    assert im_rs.shape == target_shape, im_rs.shape
    return im_rs


def augment_im(im, target_shape):
    return resize_with_padding(random_time_mask(random_freq_mask(random_zoom(random_roll(im)))), target_shape)

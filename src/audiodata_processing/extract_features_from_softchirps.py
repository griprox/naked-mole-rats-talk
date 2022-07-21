from librosa.core import piptrack
import numpy as np
import librosa
import pandas as pd


def trace_im_to_line(im):
    """ Transforms trace image into 2-d array of coordinates """
    y, x = np.where(im > 0)
    return np.array(list(zip(x, y)))


def polyfit_with_fixed_points(n, x, y, xf=[], yf=[]):
    """ Fits polynom of order n to (x,y) such that (xf, yf) are on it """
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[ : n + 1]

 
def parabola_approx(line, freq_mult):
    """ Finds parameters of parabolic approximation of the trace (as 2-d line)"""
    def _center(line):
        arg = np.argmin(line[:, 1])
        return line[arg]
    
    x, y = line[:,0], line[:,1]
    cx, cy = _center(line)
    poly = polyfit_with_fixed_points(2, x,y, )
    a = poly[2]
    b = poly[1] / (2 * a)
    c = poly[0] - poly[1] ** 2 / (4 * poly[2])
    c_freq = (128 - c) * freq_mult 
    return a, c_freq


def compute_slope_and_freq(trace, freq_mult):
    """ Computes slope and frequency of the trace (as image)"""
    line = trace_im_to_line(trace)
    slope, freq = parabola_approx(line, freq_mult)
    return slope, freq


def compute_pitch(s_npy, win_length = 220):
    """ Computes pitch of wav sound """
    pitches, magnitudes = piptrack(s_npy, sr = 22050, fmin= 2000, fmax=8000, win_length = win_length)
    bins = np.arange(pitches.shape[0])
    mean_pitch = np.mean((np.mean(pitches * magnitudes, 1)) * bins)
    return mean_pitch


def compute_amplitude(s_npy, win_length = 220):
    """ Compute average amplitude of wav """
    windows = [s_npy[i : i + win_length] for i in range(0, len(s_npy) + 1 - win_length, win_length)]
    amps = [np.max(w) - np.min(w) for w in windows]
    return np.mean(amps)


def compute_wiener_ent(s_npy, win_length = 220):
    """ Computes Wiener entropy of wav """
    ent = librosa.feature.spectral_flatness(y = s_npy, win_length = win_length)
    return np.sum(ent)


def compute_zero_crossing(s_npy,):
    """ Computes average zero crossing ratio of wav """
    return np.mean(librosa.zero_crossings(s_npy, pad=False))


def compute_duration(t, s_npy):
    """Computes duration of the trace (as image). Requires wav sound. """
    x_coverage = np.where(t > 0)[1]
    total_length = s_npy.shape[0] / 22050
    ratio = (np.max(x_coverage) - np.min(x_coverage)) / t.shape[1]
    return ratio * total_length


def compute_height(t, freq_mult):
    """ Computes hieght of the trace (as image) in Hz """
    y_coverage = np.where(t > 0)[0]
    return (np.max(y_coverage) - np.min(y_coverage)) * freq_mult


def compute_asymmetry(t, freq_mult):
    """ Computes asymmetry between height of left and right ends of the trace (as image) in Hz """
    y_coverage, x_coverage = np.where(t > 0)
    argmin_x = np.argmin(x_coverage)
    argmax_x = np.argmax(x_coverage)
    y_val_left = y_coverage[argmin_x]
    y_val_right = y_coverage[argmax_x]
    return (y_val_left - y_val_right) * freq_mult


def compute_all_features(traces, wavs, freq_mult):
    """ Computes all the features above for list of traces and corresponding wavs """
    dataset = []
    inds_skipped = []
    for ind in range(len(traces)):
        t = traces[ind]
        s = wavs[ind]
        try:
            slope, freq = compute_slope_and_freq(t, freq_mult)
            pitch = compute_pitch(s)
            amp = compute_amplitude(s)
            ent = compute_wiener_ent(s)
            zero_cr = compute_zero_crossing(s)
            dur = compute_duration(t, s)
            height = compute_height(t, freq_mult)
            asym = compute_asymmetry(t, freq_mult)
            dataset.append((slope, freq, pitch, amp, ent, zero_cr, dur, height, asym))
        except:
            inds_skipped.append(ind)
        
    return pd.DataFrame(dataset, columns = ['slope', 'frequency', 'pitch', 'amplitude', 
                                            'wiener_entropy', 'zero_crossings', 'duration', 
                                            'height', 'asymmetry']), inds_skipped
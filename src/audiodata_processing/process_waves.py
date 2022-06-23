import noisereduce as nr
from scipy import signal
import numpy as np
import librosa


def butter_highpass(cutoff, sr, order=5):
    """ Auxiliary function for filtering """
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, sr=22050, order=5):
    """ Filters recording with the highpass filter """
    if cutoff is None or cutoff <= 0:
        return data
    b, a = butter_highpass(cutoff, sr, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def add_noise(rec, mean=0.005):
    """ Adds random gaussian noise to the recording """
    if mean is None:
        return rec
    noise = np.random.randn(len(rec))
    return mean * noise + rec


def stretch(data, rate=1):
    """ Stretches the recording"""
    if rate is None:
        return data
    data = librosa.effects.time_stretch(data, rate)
    return data


def denoise_recording(rec, sr=22050, stationary=False, n_fft=256, 
                      time_constant_s=2, freq_mask_smooth_hz=500, 
                      time_mask_smooth_ms=25,):
    
    rec_f = nr.reduce_noise(y=rec, sr=22050, stationary=stationary, n_fft=n_fft, 
                            time_constant_s=time_constant_s, freq_mask_smooth_hz=freq_mask_smooth_hz, 
                            time_mask_smooth_ms=time_mask_smooth_ms,
                            n_jobs=-1)
    return rec_f


def process_waves(sounds_npy, stretching_rate_lim, noise_sigma_lim, filtering_th):
    """ Performs data augmentation methods on sounds from sounds_npy"""
    sounds_array_pr = []
    inds = list(range(0, len(sounds_npy)))
    
    if stretching_rate_lim is not None:
        stretching_rates = np.random.uniform(stretching_rate_lim[0], stretching_rate_lim[1], 
                                             size = len(sounds_npy)).tolist()
    else:
        stretching_rates = [None] * len(sounds_npy)
        
    if noise_sigma_lim is not None:
        noise_sigmas = np.random.uniform(noise_sigma_lim[0], noise_sigma_lim[1],
                                         size = len(sounds_npy)).tolist()
    else:
        noise_sigmas = [None] * len(sounds_npy)

    def _process_sound(ind):
        s_pr = stretch(sounds_npy[ind], stretching_rates[ind])
        s_pr = add_noise(s_pr, noise_sigmas[ind])
        if filtering_th is not None:
            s_pr = butter_highpass_filter(s_pr, filtering_th)
        return s_pr
    sounds_npy_pr = list(map(_process_sound, inds))
    return sounds_npy_pr

        
    
from scipy.interpolate import interp2d
from scipy.signal import stft
import numpy as np
import librosa

EPSILON = 1e-12


def extract_specs_new(sounds_metadata, sounds_npy, p, sr=22050):
    target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'])
    specs_new = []
    for temp_audio, (s, e, rec_name) in zip(sounds_npy, sounds_metadata[['s', 'e', 'rec']].values):
        s, e, = int(s * sr), int(e * sr)
        t1, t2 = s/sr, e/sr
        temp_audio = np.copy(temp_audio)
        temp_audio = temp_audio - np.mean(temp_audio)
        f, t, spec = stft(temp_audio, fs=sr, nperseg=p['nperseg'], noverlap=p['noverlap'])
        t += max(0, t1)
        spec = np.log(np.abs(spec) + EPSILON)
        interp = interp2d(t, f, spec, copy=False, bounds_error=False, fill_value=-1/EPSILON)

        max_dur = 0.2
        duration = t2 - t1
        shoulder = 0.5 * (max_dur - duration)
        target_times = np.linspace(t1-shoulder, t2+shoulder, p['num_time_bins'])
        interp_spec = interp(target_times, target_freqs, assume_sorted=True)
        interp_spec -= p['spec_min_val']
        interp_spec /= (p['spec_max_val'] - p['spec_min_val'])
        interp_spec = np.clip(interp_spec, 0.0, 1.0)
        specs_new.append(np.flip(interp_spec, 0))
    return np.array(specs_new)


def extract_mfcc(sounds_list, n_mel=40,):
    """ Compute 1d vector features of mean, var and median of mfcc's taken in time axis """
    mfcc_list = []
    for sound in sounds_list:
        mfcc = librosa.feature.mfcc(y=sound, sr=22050, n_mfcc=n_mel).T
        mfcc_mean = np.mean(mfcc, 0)
        mfcc_var = np.var(mfcc, 0)
        mfcc_median = np.median(mfcc, 0)
        mfcc_list.append(np.concatenate([mfcc_mean, mfcc_var, mfcc_median]))
    return np.array(mfcc_list)


def extract_specs(sounds_list, nfft, amp_to_db=True):
    """ Computes regular spectrograms """
    specs_list = []
    for sound in sounds_list:
        spec = np.abs(librosa.stft(sound, n_fft=nfft))
        spec = np.flip(spec, 0)
        if amp_to_db:
            spec = librosa.amplitude_to_db(spec, )
        specs_list.append(spec)
    return specs_list


def extract_melspecs(sounds_list, sr, nfft, n_mel):
    """ Computes mel spectrograms """
    mel_basis = librosa.filters.mel(sr, n_fft=nfft, n_mels=n_mel)
    specs_list = []
    for sound in sounds_list:
        spec, _ = librosa.core.spectrum._spectrogram(y=sound, n_fft=nfft, hop_length=int(nfft / 2), power=1)
        spec = np.log(np.dot(mel_basis, spec))
        specs_list.append(spec)
    return specs_list

    


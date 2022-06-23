import numpy as np
import warnings
from scipy.signal import stft
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

EPSILON = 1e-12


def create_specs_new(sounds_metadata, recs_dict, p, sr=22050):
    target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'])
    specs_new = []
    for s, e, rec_name in sounds_metadata[['s', 'e', 'rec']].values:
        t1, t2 = s/sr, e/sr
        audio = recs_dict[rec_name]
        s1, s2 = int(round(t1*sr)), int(round(t2*sr))
        temp = min(len(audio), s2) - max(0,s1)

        temp_audio = audio[max(0,s1):min(len(audio),s2)]
        temp_audio = temp_audio - np.mean(temp_audio)
        #print('shape', np.shape(temp_audio))
        f, t, spec = stft(temp_audio, fs=sr, nperseg=p['nperseg'], noverlap=p['noverlap'])

        t += max(0,t1)
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

def grid_plot(specs, gap=3, vmin=0.0, vmax=1.0, ax=None, save_and_close=False, filename='temp.pdf'):
    plt.figure(figsize=(12, 12))
    if type(gap) == type(4):
        gap = (gap,gap)
    try:
        a, b, c, d = specs.shape
    except:
        print("Invalid shape:", specs.shape, "Should have 4 dimensions.")
        quit()
    dx, dy = d+gap[1], c+gap[0]
    height = a*c + (a-1)*gap[0]
    width = b*d + (b-1)*gap[1]
    img = np.zeros((height, width))
    for j in range(a):
        for i in range(b):
            img[j*dy:j*dy+c,i*dx:i*dx+d] = specs[-j-1,i]
    for i in range(1,b):
        img[:,i*dx-gap[1]:i*dx] = np.nan
    for j in range(1,a):
        img[j*dy-gap[0]:j*dy,:] = np.nan
    if ax is None:
        ax = plt.gca()
    plt.imshow(img, aspect='equal', interpolation='none', )
    plt.clim(0, 1)
    ax.axis('off')
    
    if save_and_close:
        plt.tight_layout()
        #plt.savefig(filename)
        plt.close('all')
        
from src.audiodata_processing.process_waves import butter_highpass_filter
from src.metadata_processing.filters import filter_recs_metadata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re



def load_info_file(path):
    """ Loads files with rat information, it should have fixed format """
    info = pd.read_csv(path,)
    info = info.rename(index=str, columns={'average_weight': 'weight', 'DOB': 'dob'})
    for col in info.columns:
        info[col] = info[col].apply(lambda x: np.nan if x == 'unknown' else x)
    info.colony = info.colony.apply(lambda x: x.lower() if isinstance(x, str) else x)
    info['rank'] = info['rank'].fillna('na')
    return info


def load_recs_dict(recs_metadata, preloaded_recs_dict={}):
    """ Makes dictionary {rec_name : rec_wav} for recs from recs_metadata """
    recs_dict = {}
    for rec_name in recs_metadata['name'].unique():
        if rec_name in preloaded_recs_dict:
            rec = preloaded_recs_dict[rec_name]
        else:
            rec_path = recs_metadata[recs_metadata['name'] == rec_name]['path'].iloc[0]
            rec = np.load(rec_path + rec_name)
        recs_dict[rec_name] = rec
    return recs_dict


def load_recordings_metadata(path_to_recordings_metadata, recordings_metadata_name,
                             dates, colonies, experiments, stages, preloaded_recs_dict={}, do_load_recs_dict=True):
    recs_metadata = pd.read_csv(path_to_recordings_metadata + recordings_metadata_name)
    mask = filter_recs_metadata(recs_metadata, dates, colonies, stages, experiments)
    recs_metadata_to_use = recs_metadata[mask].reset_index(drop=True)
    if do_load_recs_dict:
        recs_dict = load_recs_dict(recs_metadata_to_use, preloaded_recs_dict)
    else:
        recs_dict = preloaded_recs_dict
    print('Using %d recordings' % len(recs_metadata_to_use))
    return recs_metadata_to_use, recs_dict


def load_sounds(sounds_metadata, recs_dict, noisy_sampling=True, sr=22050,  timestamps='sec'):
    """ Loads all sounds from sounds_metadata """
    sounds_npy = []
    if timestamps == 'int':
        s_ints = sounds_metadata['s']
        e_ints = sounds_metadata['e']
    elif timestamps == 'sec':
        s_ints = sounds_metadata['s'].apply(lambda x : int(sr * x))
        e_ints = sounds_metadata['e'].apply(lambda x : int(sr * x))
    else:
        raise ValueError('Unknown parameter timestamps=%s' % timestamps)
    if noisy_sampling:
        max_e_shifts = sounds_metadata['rec'].apply(lambda x : len(recs_dict[x])) - e_ints - 1
        diffs = e_ints - s_ints
        s_shifts = np.minimum(diffs.apply(lambda x : np.random.randint(0, x / 6)), s_ints)
        e_shifts = np.minimum(diffs.apply(lambda x : np.random.randint(0, x / 6)), max_e_shifts)
        s_ints = s_ints - s_shifts
        e_ints = e_ints + e_shifts
    for ind in range(len(sounds_metadata)):
        rec_name = sounds_metadata['rec'].iloc[ind]
        rec = recs_dict[rec_name]
        s_int, e_int = s_ints.iloc[ind], e_ints.iloc[ind]
        sounds_npy.append(rec[s_int : e_int])
    return sounds_npy


def load_traces_wavs_and_metadata(recs_metadata, info, recs_dict = None, filter_wavs=True):
    """ Loads traces and sounds for recordings in recs metadata. 
        If recording_dict is provided (dict{rec_name : rec_wav}) recordings will taken from it,
        otherwise, the rec_dict will be loaded"""
    traces, wavs, sounds_metadata,  = [], [], []
    if recs_dict is None:
        recs_dict = load_recs_dict(recs_metadata)
        
    for (rec_name, rec_path, colony, 
         date, ratids, exp) in recs_metadata[['name', 'path', 'colony', 'date', 'ratids', 'experiment']].values:

        path_to_traces = rec_path.replace('recordings', 'traces')
        rec_traces = [t for t in os.listdir(path_to_traces) if rec_name[:-3] in t]
        for tr_name in rec_traces:
            tr = plt.imread(path_to_traces + tr_name)
            traces.append(tr)
            s, e = map(lambda x: int(22050 * float(x)), 
                       re.findall('[0-9]+\.[0-9]+\_[0-9]+\.[0-9]+', tr_name)[0].split('_'))
            cl = re.findall('_[a-z]+_', tr_name)[0][1 : -1]
            wav = recs_dict[rec_name][s : e]
            if filter_wavs:
                wav = butter_highpass_filter(wav, 3000, 22050)
            wavs.append(wav)
            sounds_metadata.append((rec_name, colony, date, ratids, exp, s / 22050, e / 22050, cl))
    
    sounds_metadata = pd.DataFrame(sounds_metadata, columns = ['rec', 'colony', 'date', 'ratids', 
                                                               'experiment', 's', 'e', 'cl'])
    return traces, wavs, sounds_metadata
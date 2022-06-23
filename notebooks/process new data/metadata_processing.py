import pandas as pd
import numpy as np
import librosa
import os

def overwrite_with_npy(path, sr = 22050):
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

def add_column_to_all_recordings(recs_metadata, additional_parameters = {}):
    
    """ This function should be used if all recordings have other
        metainfo paramter(s) that are same for all recordings """
    
    new_recs_metadata = []
    for mdt in recs_metadata:
        new_recs_metadata.append(dict(mdt))
        for k, v in additional_parameters.items():
            new_recs_metadata[-1][k] = v
    return new_recs_metadata    


def compute_mask(recs_metadata, dates = 'all', colonies = 'all', stages = 'all', experiments = 'all'):
    
    """ Each filter should be either 'all' or list of acceptable values 
        Returns filtered version of recordings metadata"""
    
    if dates != 'all':
        date_mask = np.zeros(recs_metadata.shape[0], dtype = 'bool')
        for d in dates:
            date_mask[recs_metadata['date'] == d] = True
    else:
        date_mask = np.ones(recs_metadata.shape[0], dtype = 'bool')
        
    if colonies != 'all':
        col_mask = np.zeros(recs_metadata.shape[0], dtype = 'bool')
        for col in colonies:
            col_mask[recs_metadata['colony'] == col] = True
    else:
        col_mask = np.ones(recs_metadata.shape[0], dtype = 'bool')

    if stages != 'all':
        stage_mask = np.zeros(recs_metadata.shape[0], dtype = 'bool')
        for st in stages:
            stage_mask[recs_metadata['processing stage'] == st] = True
    else:
        stage_mask = np.ones(recs_metadata.shape[0], dtype = 'bool')

    if experiments != 'all':
        exp_mask = np.zeros(recs_metadata.shape[0], dtype = 'bool')
        exp_mask = recs_metadata['experiment'].apply(lambda e_list: max([e in experiments for e in e_list]))
    else:
        exp_mask = np.ones(recs_metadata.shape[0], dtype = 'bool')
    
    return date_mask * col_mask * stage_mask * exp_mask

def update_metadata(old_recs_metadata, new_recs_metadata, additional_parameters):
    
    """ This function compares old and new metadata files 
        and returns their concatenation """
    
    if len(old_recs_metadata) == 0:
        return new_recs_metadata
    old_recs_metadata_copy = pd.DataFrame(np.array(old_recs_metadata), columns = old_recs_metadata.columns)
    new_recs_metadata_copy = pd.DataFrame(np.array(new_recs_metadata), columns = new_recs_metadata.columns)
    
    new_columns = [c for c in new_recs_metadata.columns if c not in old_recs_metadata.columns]
    unfilled_columns = [c for c in old_recs_metadata.columns if c not in new_recs_metadata.columns]
    print('New recordings do not have values for following metadata columns:')
    print(unfilled_columns)
    print('Current metadata file does not have these columns:')
    print(new_columns)
        
    for c in new_columns:
        old_recs_metadata_copy[c] = [np.nan for _ in range(len(old_recs_metadata_copy))]
    for c in unfilled_columns:
        new_recs_metadata_copy[c] = [np.nan for _ in range(len(new_recs_metadata_copy))]
    
    repeated_entries = set(old_recs_metadata['name']) & set(new_recs_metadata['name'])
    inds_to_drop = []
    if len(repeated_entries):
        print('%d/%d recordings are already in the metadata' % (len(repeated_entries), len(new_recs_metadata)))
    for rec_name in repeated_entries:
        ind_old = np.where(old_recs_metadata_copy['name'] == rec_name)[0][0]
        ind_new = np.where(new_recs_metadata_copy['name'] == rec_name)[0][0]
        inds_to_drop.append(ind_new)
        exp_old = old_recs_metadata_copy['experiment'].iloc[ind_old]
        exp_new = new_recs_metadata_copy['experiment'].iloc[ind_new]
        exp_updated = sorted(list(set(exp_old + exp_new)))
        old_recs_metadata_copy['experiment'].iloc[ind_old] = exp_updated
    new_recs_metadata_copy = new_recs_metadata_copy.drop(inds_to_drop)
    return pd.concat([old_recs_metadata_copy, new_recs_metadata_copy], 0).reset_index(drop = True)


def generate_sounds_metadata(recs_metadata, 
                             columns_to_copy = ['colony', 'ratids', 'date', 'experiment']):
    
    """ Generates metadata table for all sounds from all recordings that are not fresh 
        Copies columns_to_copy columns from recordings metadata into sounds_metadata  """
    
    processed_recs = recs_metadata[recs_metadata['processing stage'] != 'fresh']
    sounds_metadata= []
    for p, n in processed_recs[['path', 'name']].values:
        rec_row = recs_metadata[recs_metadata['name'] == n]
        df_txt = pd.read_csv(p + n[:-3] + 'txt', sep = '\t')
        df_txt['rec'] = n
        for c in columns_to_copy:
            df_txt[c] = rec_row[c].iloc[0]
        sounds_metadata.append(df_txt)
    sounds_metadata = pd.concat(sounds_metadata, 0)
    return sounds_metadata

def load_recordings_for_sounds(sounds_metadata, recs_metadata):
    
    """ Loads recordings for which there are sounds in sounds_metadta into 
        dictionary dict[rec name] = rec.npy """
    
    recs_dict = {}
    for rn in set(sounds_metadata['rec']):
        p = recs_metadata[recs_metadata['name'] == rn]['path'].iloc[0]
        recs_dict[rn] = np.load(p + rn)
    return recs_dict
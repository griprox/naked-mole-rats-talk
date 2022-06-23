from src.util import date_to_datetime
import numpy as np
import pandas as pd
import os


def compute_epochs(sounds_metadata, EPOCHS):
    """ Generates sounds_metadata with additional epochs column """
    def _get_epoch(dt, epochs_dict):
        if epochs_dict is None:
            return '?'
        for s, e in epochs_dict:
            if date_to_datetime(s) <= dt <= date_to_datetime(e):
                return epochs_dict[(s, e)]
    epochs = []
    for ind in range(len(sounds_metadata)):
        col = sounds_metadata['colony'].iloc[ind]
        date_dt = date_to_datetime(sounds_metadata['date'].iloc[ind])
        epochs.append(_get_epoch(date_dt, EPOCHS[col]))
    sounds_metadata['epoch'] = epochs
    return sounds_metadata


def generate_sounds_metadata(recs_metadata, classes_to_drop, max_sounds_per_class, min_sound_length, max_sound_length,
                             columns_to_copy=['colony', 'ratids', 'date', 'experiment'],
                             min_sound_occurrences=0, sr=22050, print_counts=True):
    # Create sounds metadata by loading txts
    sounds_metadata = []
    for p, n in recs_metadata[['path', 'name']].values:
        rec_row = recs_metadata[recs_metadata['name'] == n]
        if os.path.isfile(p + n[:-3] + 'txt'):
            df_txt = pd.read_csv(p + n[:-3] + 'txt', sep='\t')
            df_txt['rec'] = n
            for c in columns_to_copy:
                df_txt[c] = rec_row[c].iloc[0]
            sounds_metadata.append(df_txt[~df_txt['cl'].isin(classes_to_drop)])
    sounds_metadata = pd.concat(sounds_metadata, 0).reset_index(drop=True)

    # Keep class sizes within limitsmerge_recs_metadata
    for cl, max_num in max_sounds_per_class.items():
        cl_inds = np.where(sounds_metadata['cl'].isin([cl]))[0]
        np.random.shuffle(cl_inds)
        sounds_metadata = sounds_metadata.drop(cl_inds[max_num:]).reset_index(drop=True)
    # Delete underrepresented classes
    underrepresented_classes = []
    for cl, n_occurrences in dict(sounds_metadata['cl'].value_counts()).items():
        if n_occurrences < min_sound_occurrences:
            underrepresented_classes.append(cl)
    underrepresented_classes_mask = sounds_metadata['cl'].isin(underrepresented_classes)
    sounds_metadata = sounds_metadata[~underrepresented_classes_mask].reset_index(drop=True)
    # Delete sounds that are too short or too long (probably bad splits)
    lengths = sr * (sounds_metadata['e'] - sounds_metadata['s'])
    good_length_mask = (lengths >= min_sound_length) & (lengths <= max_sound_length)
    sounds_metadata = sounds_metadata[good_length_mask].reset_index(drop=True)
    if print_counts:
        print(sounds_metadata['cl'].value_counts())
    return sounds_metadata


def split_in_train_and_test(sounds_metadata, train_ratio=0.9, print_counts=True, random_seed=None):
    all_recs = sounds_metadata['rec'].unique()
    total_recordings = len(all_recs)
    if random_seed is not None:
        print('Set seed:', random_seed, 'total', total_recordings)
        np.random.seed(random_seed)
    recs_ids = np.arange(total_recordings, dtype='int')
    np.random.shuffle(recs_ids)
    N_train_recs = int(train_ratio * total_recordings)
    train_recs_ids = recs_ids[: N_train_recs]
    test_recs_ids = recs_ids[N_train_recs:]
    train_recs = all_recs[train_recs_ids]
    test_recs = all_recs[test_recs_ids]
    sounds_metadata_train = sounds_metadata[sounds_metadata['rec'].isin(train_recs)].reset_index(drop=True)
    sounds_metadata_test = sounds_metadata[sounds_metadata['rec'].isin(test_recs)].reset_index(drop=True)
    if print_counts:
        print('Using %d recordings (%d sounds)for train and' % (len(train_recs), len(sounds_metadata_train)),
              ' %d recordings (%d sounds)for test' % (len(test_recs), len(sounds_metadata_test)))
        print()
        print('############ Classes in train:############ \n')
        print(sounds_metadata_train['cl'].value_counts())
        print('############ Classes in test: ############ \n')
        print(sounds_metadata_test['cl'].value_counts())
    return sounds_metadata_train, sounds_metadata_test


def extend_sounds_metadata(sounds_metadata, info):
    """ Extends sounds metadata with ratinfo """
    each_rat_info = {}
    for ratid in sounds_metadata['ratids'].unique():
        if '_' in ratid or 'whole' in ratid:
            w = s = r = db = bl = bl2 = age = np.nan
        elif int(ratid) not in info['ID'].values:
            print('No info for %s' % ratid)
            w = s = r = db = bl = bl2 = age = np.nan
        else:
            ind_in_info = np.where(info['ID'] == int(ratid))[0][0]
            w, s, r, db, bl, bl2 = info[['weight', 'sex', 'rank', 'dob', 'body length', 'body length2']].iloc[ind_in_info]
        each_rat_info[ratid] =  (w, s, r, db, bl, bl2)
        
    columns_to_add = []
    for ind in range(sounds_metadata.shape[0]):
        w, s, r, db, bl, bl2 = each_rat_info[sounds_metadata['ratids'].iloc[ind]]
        date = sounds_metadata['date'].iloc[ind]
        age = (date_to_datetime(date) - date_to_datetime(db)).days if (db is not np.nan) else np.nan
        columns_to_add.append((w, s, r, db, age, bl, bl2))
        
    columns_to_add = pd.DataFrame(columns_to_add, columns = ['weight', 'sex', 'rank', 
                                                             'dob', 'age', 'bodylength', 'bodylength2'])
    return pd.concat([sounds_metadata.reset_index(drop = True), columns_to_add], 1)


def make_fixed_size_sounds(sounds_metadata, resolution=1024, step=512, sr=22050):
    """ Changes timesstamps s.t. sounds are all of the same size """
    sounds_metadata_split = []

    s_ints = sounds_metadata['s'].apply(lambda x: int(sr * x))
    e_ints = sounds_metadata['e'].apply(lambda x: int(sr * x))
    sizes = e_ints - s_ints

    s_col_ind = list(sounds_metadata.columns).index('s')
    e_col_ind = list(sounds_metadata.columns).index('e')

    for ind in range(len(sounds_metadata)):
        s_int, e_int, size = s_ints.iloc[ind], e_ints.iloc[ind], sizes.iloc[ind]
        parts_in_sound = size // resolution
        useless_space = size - parts_in_sound * resolution

        s_int_new = int(s_int + useless_space // 2)
        e_int_new = int(e_int - useless_space // 2)

        for s_p in range(s_int_new, e_int_new + 1 - resolution, step):
            e_p = s_p + resolution
            row = list(sounds_metadata.iloc[ind])
            row[s_col_ind] = s_p
            row[e_col_ind] = e_p
            sounds_metadata_split.append(tuple(row))
    sounds_metadata_split = pd.DataFrame(sounds_metadata_split, columns=sounds_metadata.columns)
    return sounds_metadata_split

from src.util import date_to_datetime
from datetime import datetime
import pandas as pd
import numpy as np


def filter_recs_metadata(recs_metadata, dates='all', colonies='all', stages='all', experiments='all'):
    """ Filters recs_metadata, returns mask """
    if dates == 'all':
        mask_d = np.ones(len(recs_metadata), dtype = 'bool')
    else:
        dates_transformed = recs_metadata['date'].apply(date_to_datetime)
        d_min, d_max = date_to_datetime(dates[0]), date_to_datetime(dates[1])
        mask_d = dates_transformed.apply(lambda x: d_min <= x <= d_max)
    if colonies == 'all':
        mask_c = np.ones(len(recs_metadata), dtype = 'bool')
    else:
        mask_c = recs_metadata['colony'].isin(colonies)
    if stages == 'all':
        mask_s = np.ones(recs_metadata.shape[0], dtype = 'bool')
    else:
        mask_s = recs_metadata['processing stage'].isin(stages)
    if experiments == 'all':
        mask_e = np.ones(recs_metadata.shape[0], dtype = 'bool')
    else:
        mask_e = recs_metadata['experiment'].apply(lambda es: max([e in experiments for e in es.split(';')]))
    return np.array(mask_d * mask_c * mask_s * mask_e)


def filter_sounds_metadata_extended(sounds_metadata_extended, ratids='all', excludeids=None, types='all',
                                    colonies='all', weights='all', sex='all', age='all', ranks='all',
                                    dates='all', bodylength='all', bodylength2='all', sounds_per_rat='all'):
    """ Filters sounds_metadata extended with ratinfo, returns mask """
    if ratids is 'all':
        mask_ids = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_ids = sounds_metadata_extended['ratids'].isin(ratids)
    if excludeids is None:
        mask_exclude = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_exclude = np.ones(len(sounds_metadata_extended), dtype = 'bool')
        for i in excludeids:
            mask_exclude = np.logical_and(mask_exclude, sounds_metadata_extended['ratids'] != i)
    if colonies is 'all':
        mask_c = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_c = sounds_metadata_extended['colony'].isin(colonies)
    if sex is 'all':
        mask_s = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_s = sounds_metadata_extended['sex'].isin(sex)
    if ranks is 'all':
        mask_r = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_r = sounds_metadata_extended['rank'].isin(ranks)
    if weights is 'all':
        mask_w = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_w = sounds_metadata_extended['weight'].apply(lambda x : weights[0] <= x <= weights[1])
    if age is 'all':
        mask_a = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_a = sounds_metadata_extended['age'].apply(lambda x : age[0] <= x <= age[1])
    if bodylength is 'all':
        mask_bl = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_bl = sounds_metadata_extended['bodylength'].apply(lambda x : bodylength[0] <= x <= bodylength[1])
    if bodylength2 is 'all':
        mask_bl2 = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_bl2 = sounds_metadata_extended['bodylength2'].apply(lambda x : bodylength2[0] <= x <= bodylength2[1])
    if dates is 'all':
        mask_d = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        date_to_dt = lambda x: datetime(*reversed(list(map(int, x.replace('-', '.').split('.')))))
        dates_transformed = sounds_metadata_extended['date'].apply(date_to_dt)
        d_min, d_max = date_to_dt(dates[0]), date_to_dt(dates[1])
        mask_d = dates_transformed.apply(lambda x: d_min <= x <= d_max)
    if types is 'all':
        mask_types = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_types = np.zeros(len(sounds_metadata_extended), dtype = 'bool')
        if 'single' in types:
            mask_types[sounds_metadata_extended['ratids'].apply(lambda x : len(x) <= 5)] = 1
        if 'pair' in types:
            mask_types[sounds_metadata_extended['ratids'].apply(lambda x : len(x) > 5)] = 1
    mask = mask_r & mask_ids & mask_exclude & mask_types & mask_c
    mask = mask & mask_w & mask_a & mask_s & mask_d & mask_bl & mask_bl2
    if sounds_per_rat is 'all':
        mask_select = np.ones(len(sounds_metadata_extended), dtype = 'bool')
    else:
        mask_select = np.ones(len(sounds_metadata_extended), dtype = 'bool')
        th_low, th_up = sounds_per_rat
        id_inds = {}
        for rat_id in set(sounds_metadata_extended[mask]['ratids']):
            id_inds = np.where((sounds_metadata_extended['ratids'] == rat_id) & mask)[0]
            if len(id_inds) < th_low:
                mask_select[id_inds] = False
            elif len(id_inds) > th_up:
                np.random.shuffle(id_inds)
                mask_select[id_inds[th_up:]] = False
                mask_select[id_inds[th_up:]] = False
    mask = mask & mask_select
    return np.array(mask)

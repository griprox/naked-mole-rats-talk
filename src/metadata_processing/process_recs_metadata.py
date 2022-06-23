import pandas as pd
import numpy as np


def add_columns_to_recs_metadata(recs_metadata: pd.DataFrame, additional_parameters: dict):
    """ Adds new column column_name: value for each item in the additional_parameters dict
    This function should be used if all recordings have metadata column(s) which are same for all recordings """
    new_recs_metadata = pd.DataFrame(recs_metadata)
    for col_name, val in additional_parameters.items():
        new_recs_metadata[col_name] = len(new_recs_metadata) * [val]
    return new_recs_metadata    


def merge_recs_metadata(old_recs_metadata: pd.DataFrame, new_recs_metadata: pd.DataFrame):
    """ This function compares old and new metadata files 
        and returns their concatenation """
    if len(old_recs_metadata) == 0:
        return new_recs_metadata
    old_recs_metadata_copy = pd.DataFrame(np.array(old_recs_metadata), columns=old_recs_metadata.columns)
    new_recs_metadata_copy = pd.DataFrame(np.array(new_recs_metadata), columns=new_recs_metadata.columns)
    
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
        inds_to_drop.append(ind_old)
        exp_old = str(old_recs_metadata_copy['experiment'].iloc[ind_old])
        exp_new = str(new_recs_metadata_copy['experiment'].iloc[ind_new])
        
        if exp_old == exp_new:
            exp_updated = exp_old
        else:
            exp_updated = exp_old + ';' + exp_new
        old_recs_metadata_copy['experiment'].iloc[ind_old] = exp_updated
    old_recs_metadata_copy = old_recs_metadata_copy.drop(inds_to_drop)
    return pd.concat([old_recs_metadata_copy, new_recs_metadata_copy], 0).reset_index(drop=True)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing_old.process_softchirp_features import choose_features
from src.util import get_ordered_y, get_grouping_indexes, get_balancing_indexes
from src.visualization.plotting_utils import plot_confusion_matrix
import matplotlib.pyplot as plt

def make_bins(column, binsize = 100):
    column_new = column.apply(lambda x : binsize * int(x / binsize))
    return column_new

def generate_yi_folds(keys_in_yi, cv_n_folds):
    folds = np.array_split(keys_in_yi, cv_n_folds)
    folds = [list(f) for f in folds if len(f) > 0]
    return folds

def compute_grouping_keys_folds(sounds_metadata_data, classify_by, group_by, cv_n_folds = 5):
    folds_in_y = {}
    if group_by == None:
        grouping_col = pd.Series(np.arange(0, len(sounds_metadata_data)), index = sounds_metadata_data.index)
    else:
        grouping_col = sounds_metadata_data[group_by]
                                 
    for yi in sounds_metadata_data[classify_by].unique():
        yi_mask = sounds_metadata_data[classify_by] == yi
        keys_in_yi = grouping_col[yi_mask].unique().tolist()
        np.random.shuffle(keys_in_yi)
        folds_in_y[yi] = generate_yi_folds(keys_in_yi, cv_n_folds)
    return folds_in_y

def prepare_data(features, sounds_metadata, mask, mask_foster,
                 classify_by, features_to_classify, order, balance = True):
    
    y_unique = get_ordered_y(sounds_metadata[mask][classify_by], order)
    y_unique_foster = get_ordered_y(sounds_metadata[mask | mask_foster][classify_by], order)
    y_num_data_raw = np.array(list(map(y_unique.index, sounds_metadata[mask][classify_by])))
    features_data_raw, _ = choose_features(features[mask], features_to_classify)
    sounds_metadata_data_raw = sounds_metadata[mask]
    
    if balance:
        data_indexes = get_balancing_indexes(y_num_data_raw)
        y_num_data = y_num_data_raw[data_indexes]
        features_data = features_data_raw[data_indexes]
        sounds_metadata_data = sounds_metadata_data_raw.iloc[data_indexes]
    else:
        y_num_data = y_num_data_raw
        features_data = features_data_raw
        sounds_metadata_data = sounds_metadata_data_raw
        
    
    y_foster = sounds_metadata[mask_foster][classify_by]
    y_num_foster = np.array(list(map(y_unique_foster.index, y_foster)))
    features_foster, _ = choose_features(features[mask_foster], features_to_classify)
    sounds_metadata_foster = sounds_metadata[mask_foster]
    return (y_unique, y_unique_foster, features_data, y_num_data, sounds_metadata_data,
            features_foster, y_num_foster, sounds_metadata_foster)


def get_train_test_indexes(sounds_metadata_data, folds_in_y, nfold, classify_by, group_by):
    
    train_indexes = []
    test_indexes = []
    if group_by == None:
        grouping_col = pd.Series(np.arange(0, len(sounds_metadata_data)), index = sounds_metadata_data.index)
    else:
        grouping_col = sounds_metadata_data[group_by]
        
    for yi in folds_in_y: 
        
        yi_folds = folds_in_y[yi]
        test_fold_ind = nfold % len(yi_folds) 
        if len(yi_folds) <= 1:
            print('For %s only one fold %s' %(yi, yi_folds))
        train_keys = np.concatenate(yi_folds[ : test_fold_ind] + yi_folds[test_fold_ind + 1 :]).tolist()
        test_keys = yi_folds[test_fold_ind]
        for k in train_keys:
            inds = np.where((sounds_metadata_data[classify_by] == yi) & (grouping_col == k))[0]
            train_indexes.extend(inds)
            
        for k in test_keys:
            inds = np.where((sounds_metadata_data[classify_by] == yi) & (grouping_col == k))[0]
            test_indexes.extend(inds)
    
    return train_indexes, test_indexes

def check_indexes_are_good(sounds_metadata_data, train_indexes, test_indexes, 
                           classify_by, group_by, print_info = False):
    
    train_metadata = sounds_metadata_data.iloc[train_indexes]
    test_metadata = sounds_metadata_data.iloc[test_indexes]
    if print_info:
        print('\nTRAIN SET:')
        print(dict(train_metadata[classify_by].value_counts()))
        print(dict(train_metadata[group_by].value_counts()))
        print('\nTEST SET:')
        print(dict(test_metadata[classify_by].value_counts()))
        print(dict(test_metadata[group_by].value_counts()))
        
    train_keys_check = sounds_metadata_data.iloc[train_indexes][group_by].unique().tolist()
    test_keys_check = sounds_metadata_data.iloc[test_indexes][group_by].unique().tolist()
    common_keys = set(train_keys_check) & set(test_keys_check)
    for k in common_keys:
        common_y = (set(train_metadata[train_metadata[group_by] == k][classify_by].unique()) &
                    set(test_metadata[test_metadata[group_by] == k][classify_by].unique()))
        assert len(common_y) == 0, 'Both train and test parts have groupping key %s for y in %s' %((k), common_y)
        
    
def classify_foster_pups(clf, sounds_metadata_foster, features_foster, y_unique, y_unique_num):
    
    foster_probs = {}   
    for ratid in sounds_metadata_foster['ratids'].unique():

        inds = np.where(sounds_metadata_foster['ratids'] == ratid)[0]
        pred = clf.predict_proba(features_foster[inds])
        probs = np.mean(pred, 0)
        predicted_column = np.argmax(pred, 1)
        
        total_sounds = len(inds)
        
        print('\nFOSTER %s:' % ratid)
        print('Accuracy is %.3f' % accuracy_score(predicted_column, y_unique_num[inds]))
        for i, prob in enumerate(probs):
            col_sounds = np.sum(predicted_column == i)
            print('Colony %s with probability %.2f' % (y_unique[i], prob))
            print('Predicted colony %s for %d out of %d softchirps' % (y_unique[i], col_sounds, total_sounds))

        foster_probs[ratid] = probs
    return foster_probs




def classify(features, sounds_metadata, mask, mask_foster, order, 
             classify_by = 'colony', features_to_classify = 'all', group_by = 'ratids',
             balance = True, cv_n_folds = 5, analyse_fosters = True, print_info = False, cmap = plt.cm.Purples):
    
    clf = RandomForestClassifier(n_estimators = 256)
    accuracies, cms = [], []


    (y_unique, y_unique_foster, features_data, y_num_data, sounds_metadata_data,
    features_foster, y_num_foster, sounds_metadata_foster) = prepare_data(features, sounds_metadata,
                                                                          mask, mask_foster,
                                                                          classify_by, features_to_classify,
                                                                          order, balance)
    y_unique_cv = [yi for yi in y_unique if yi in sounds_metadata_data[classify_by].values]
    folds_in_y = compute_grouping_keys_folds(sounds_metadata_data, classify_by, group_by, cv_n_folds)


    for nfold in range(cv_n_folds):

        train_indexes, test_indexes = get_train_test_indexes(sounds_metadata_data, folds_in_y, nfold, 
                                                             classify_by, group_by)
        if group_by != None:
            check_indexes_are_good(sounds_metadata_data, train_indexes, test_indexes, 
                                   classify_by, group_by, print_info)
        elif print_info:
            print('Train set:', sounds_metadata_data.iloc[train_indexes][classify_by].value_counts())
            print('Train set:', sounds_metadata_data.iloc[test_indexes][classify_by].value_counts())

        X_train, y_train = features_data[train_indexes], y_num_data[train_indexes]
        X_test, y_test = features_data[test_indexes], y_num_data[test_indexes]
        
        if len(set(y_train)) < len(set(y_num_data)):
            print('\nNot all labels are in training data during CV, results may be unreliable')

        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, prediction))
        cms.append(confusion_matrix(y_test, prediction))

    print('Average accuracy is %.2f' % np.mean(accuracies))
    plot_confusion_matrix(np.mean(cms, 0), y_unique_cv, (10, 10), cmap)

    clf.fit(features_data, y_num_data)

    if not analyse_fosters:
        return clf, None, y_unique, y_unique

    prediction_foster = clf.predict(features_foster)
    print('Accuracy on foster pups is %.2f' % np.mean(accuracy_score(y_num_foster, prediction_foster)))

    foster_probs = classify_foster_pups(clf, sounds_metadata_foster, features_foster, y_unique, y_num_foster)
    return clf, foster_probs, y_unique, y_unique_foster
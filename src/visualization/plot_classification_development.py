import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from src.preprocessing_old.filters import filter_sounds_metadata_extended

def plot_prediction_histogram(age, ratid, sounds_metadata_extended, grouped_labels):
    
    age_rat_mask = filter_sounds_metadata_extended(sounds_metadata_extended, ratids = [ratid], age = age)
    sounds_in_rat_age = grouped_labels[age_rat_mask]
    
    sound_info = sounds_metadata_extended[age_rat_mask][['rec', 'ratids', 'cl', 'age', 's', 'e']].reset_index(drop = True)
    table = pd.concat([sound_info, pd.DataFrame(sounds_in_rat_age, columns = ['Pred#%d' % i for i in range(1, 6)])], 1)
    table =  table.sort_values(by = 'age').reset_index(drop = True)
    
    plt.figure(figsize = (10, 10))
    _ = plt.hist(sounds_in_rat_age.flatten(), normed = True)
    _ = plt.title('Predictoins of augmented data for %s of age %s' % (ratid, age))
    return table


def plot_sounds_from_table(table, sounds_to_take, subplots_dimensions):
    plt.figure(figsize = (9, 9))
    n_suplots = len(sounds_to_take)
    print(table.iloc[sounds_to_take])
    for i, ind in enumerate(sounds_to_take):

        plt.subplot(*subplots_dimensions, i + 1)
        _ = plt.title('sound % d' % ind)
        labels = table.iloc[ind][['Pred#%d' % i for i in range(5)]].values.tolist()
        plt.hist(labels)
        plt.yticks([0, 1, 2, 3, 4, 5])
        

        
def plot_sound_probability_development(labels, sounds_metadata_extended, 
                                       ratid, all_classes, age_binning = 30, sound = 'softchirp',):
    sound_index = all_classes.index(sound)
    rat_mask = sounds_metadata_extended['ratids'] == ratid
    rat_ages_binned = sounds_metadata_extended[rat_mask]['age'].apply(lambda x : age_binning * int(x//age_binning))
    rat_ages_binned_values = sorted(rat_ages_binned.unique())
    sounds_probs = []
    for age in rat_ages_binned_values:
        rat_age_mask = filter_sounds_metadata_extended(sounds_metadata_extended, 
                                                       ratids = [ratid],
                                                       age = (age, age + age_binning - 0.5))
        rat_age_sounds = labels[rat_age_mask]
        sound_prob = np.mean(rat_age_sounds == sound)
        sounds_probs.append(sound_prob)        
    plt.plot(rat_ages_binned_values, sounds_probs, marker = 'X', label = ratid)
    
def plot_matrix_for_age(labels, sounds_metadata_extended, all_classes, age, ):
    conf_matrix = []
    pups_ids = sounds_metadata_extended['ratids'].unique()
    
    for ratid in pups_ids:
        bc = np.zeros(len(all_classes))
        age_rat_mask = filter_sounds_metadata_extended(sounds_metadata_extended, ratids = [ratid], age = age)
        for cl in labels[age_rat_mask]:
            bc[all_classes.index(cl)] += 1
        conf_matrix.append(bc / bc.sum())
    
    
    fig, ax = plt.subplots(figsize = (9, 9))
    cm = np.array(conf_matrix)
    im = plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
    _ = ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]),
           xticklabels = all_classes, yticklabels = pups_ids,
           title = 'Sounds distribution at age %s' % age,
           ylabel = 'Pup',
           xlabel = 'Sound type')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.colorbar(im)
    

def plot_pup_accent_development(ratid, age_groups, foster_predictions, y_unique,
                                color_dict = None, figsize = (12, 5)):
    plt.figure(figsize = figsize)
    rat_mask = foster_predictions['ratids'] == ratid
    results = {c : [None for _ in age_groups] for c in y_unique}
    for ind, a in enumerate(age_groups):
        rat_age_mask = rat_mask * (foster_predictions['age'].apply(lambda x : a[0] <= x <= a[1]))
        rat_age_predictions = foster_predictions[rat_age_mask]
        rat_predictions_colony = np.bincount(np.argmax(rat_age_predictions[y_unique].values, 1)) / len(rat_age_predictions)
        rat_predictions_colony = np.concatenate([rat_predictions_colony, 
                                                 np.zeros(len(y_unique) - len(rat_predictions_colony))])
        for c_ind, c in enumerate(y_unique):
            results[c][ind] = rat_predictions_colony[c_ind]
            
    if color_dict is None:
        cmap = cm.rainbow
    else:
        cmap = colors.ListedColormap([color_dict[yi] for yi in y_unique])
        
    for c in y_unique:
        plt.scatter(['%d-%d days' % (a[0], a[1]) for a in age_groups], results[c], label = c, 
                   c = [color_dict[c] for _ in results[c]], )
    _ = plt.legend(loc = 1)
    _ = plt.xlabel('Age')
    _ = plt.ylabel('P')
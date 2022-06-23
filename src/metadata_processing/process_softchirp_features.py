import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def find_outliers(df, boundaries={'slope' : (0.01, .15), 'frequency' : (2000, 8000),
                                  'duration' : (0.1, 10), 'height' : (1000, 6000) }):
    """ Finds indexes of the outliers -- rows for which at value does not lie 
        in the corresponding boundary """
    outliers = []
    for col in boundaries:
        min_val, max_val = boundaries[col]
        to_drop = list(np.where((np.array(df[col]) < min_val) | (np.array(df[col]) > max_val))[0])
        outliers.extend(to_drop)
    return outliers


def choose_features(df, features_to_take, perplexity=30):
    """ Takes columns features_to_take from the df or 
        performs dimensionality reduction technique """
    if features_to_take == 'all':
        features_to_take = list(df.columns)
    if features_to_take in ['pca', 'PCA']:
        pca = PCA(n_components = 2)
        df_pca = pca.fit_transform(StandardScaler().fit_transform(df))
        exp_var =  pca.explained_variance_ratio_
        df_new = pd.DataFrame(df_pca, columns = ['pc#1 (%.3f)' % exp_var[0], 'pc#2 (%.3f)' % exp_var[1]])
        print('Variance explained', exp_var)
    elif features_to_take in ['TSNE', 'tsne']:
        tsne = TSNE(n_components=2, perplexity =perplexity)
        df_new = pd.DataFrame(tsne.fit_transform(df), columns=['dim#1', 'dim#2'])
    elif features_to_take in ['pca3d', 'PCA3d']:
        pca = PCA(n_components=3)
        df_pca = pca.fit_transform(StandardScaler().fit_transform(df))
        exp_var =  pca.explained_variance_ratio_
        df_new = pd.DataFrame(df_pca, columns=['pc#1 (%.3f)' % exp_var[0], 'pc#2 (%.3f)' % exp_var[1],
                                               'pc#3 (%.3f)' % exp_var[2]])
        print('Variance explained', pca.explained_variance_ratio_)
    elif features_to_take in ['tsne3d', 'TSNE3d']:
        tsne = TSNE(n_components=3, perplexity=perplexity)
        df_new = pd.DataFrame(tsne.fit_transform(df), columns=['dim#1', 'dim#2', 'dim#3'])
    else:
        df_new = df[features_to_take]
    return np.array(df_new), df_new.columns

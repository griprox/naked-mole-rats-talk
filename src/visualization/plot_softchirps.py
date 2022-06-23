import numpy as np
import scipy.stats as scst

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from src.util import get_ordered_y, get_grouping_indexes
from src.visualization.plotting_utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def choose_features(df, features_to_take, perplexity = 30):
    """ Takes columns features_to_take from the df orconda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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
        tsne = TSNE(n_components = 2, perplexity = perplexity)
        df_new = pd.DataFrame(tsne.fit_transform(df), columns = ['dim#1', 'dim#2'])
    elif features_to_take in ['pca3d', 'PCA3d']:
        pca = PCA(n_components = 3)
        df_pca = pca.fit_transform(StandardScaler().fit_transform(df))
        exp_var =  pca.explained_variance_ratio_
        df_new = pd.DataFrame(df_pca, columns = ['pc#1 (%.3f)' % exp_var[0], 'pc#2 (%.3f)' % exp_var[1], 'pc#3 (%.3f)' % [2]])
        print('Variance explained', pca.explained_variance_ratio_)
    elif features_to_take in ['tsne3d', 'TSNE3d']:
        tsne = TSNE(n_components = 3, perplexity = perplexity)
        df_new = pd.DataFrame(tsne.fit_transform(df), columns = ['dim#1', 'dim#2', 'dim#3'])
    else:
        df_new = df[features_to_take]
    return np.array(df_new), df_new.columns


def plot(features, y, sounds_metadata_masked, features_to_plot, perplexity = 50,
         group_by_rat = True, group_color_column = None, plot_ellipses = True, plot_color_mean = True,
         color_dict = None, order = None, sizes_factor = 32, figsize = (14, 9)):
    
    # create figure
    fig = plt.figure(figsize = figsize)
    ax = plt.axes()

    data_to_plot, axises = choose_features(features, features_to_plot, perplexity)

    # group by rat
    if group_by_rat:
        grouping_indexes = get_grouping_indexes(sounds_metadata_masked['ratids'])
        sizes = np.array([scst.sem(data_to_plot[inds], 0) if len(inds) > 1 else np.array([.1, 1.])
                          for inds in grouping_indexes])
        data_to_plot = np.array([np.mean(data_to_plot[inds], 0) for inds in grouping_indexes])
        y_to_plot = np.array([y.iloc[inds[0]] for inds in grouping_indexes])
    else:
        data_to_plot = np.array(data_to_plot)
        y_to_plot = np.array(y)
        sizes = sizes_factor * np.ones(len(data_to_plot))
    # group color column
    if group_color_column:
        y_to_plot = [group_color_column * (yi // group_color_column) for yi in y_to_plot]
    
    # set labels and colormap
    y_unique = get_ordered_y(y_to_plot, order)   
    ticks = np.arange(0, len(y_unique))
    if color_dict is None:
        cmap = cm.rainbow
    else:
        cmap = colors.ListedColormap([color_dict[yi] for yi in y_unique])
    if len(ticks) == 1:
        norm = None    
    else:
        norm = colors.BoundaryNorm(np.arange(min(ticks)- .5, max(ticks) + .6, 1), cmap.N)
    
    # plot data
    plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], c = [y_unique.index(yi) for yi in y_to_plot],
                norm = norm, cmap = cmap, alpha = 0.8)
    # plot ellipses
    norm_apply = (lambda x: x )if norm is None else norm
    if plot_ellipses:
        for i, y_val in enumerate(y_unique):
            data_i = np.array(data_to_plot[np.array(y_to_plot) == y_val])
            mean, cov = np.mean(data_i, axis = 0), np.cov(data_i[:,0], data_i[:, 1])
            e = get_cov_ellipse(cov, mean, 2,  fc = cmap(norm_apply(i)), alpha = 0.2)
            ax.add_artist(e)
    # plot errorbars for color averages
    if plot_color_mean:
        grouping_indexes = get_grouping_indexes(y_to_plot)
        sizes_color_mean = np.array([scst.sem(data_to_plot[inds], 0) if len(inds) > 1 else np.array([.1, 1.])
                          for inds in grouping_indexes])
        data_color_mean = np.array([np.mean(data_to_plot[inds], 0) for inds in grouping_indexes])
        y_color_mean = np.array([y.iloc[inds[0]] for inds in grouping_indexes])
        
        
        
        
        #data_color_mean, y_color_mean, sizes_color_mean = average_by_column(data_to_plot, y_to_plot, y_to_plot)
        plt.scatter(data_color_mean[:, 0],data_color_mean[:, 1], 
                    norm = norm, cmap = cmap, s = sizes_factor,
                    c = [y_unique.index(yi) for yi in y_color_mean], marker = ',', alpha = 1)
        
        plot_error_bars(data_color_mean, [y_unique.index(yi) for yi in y_color_mean],
                        sizes_color_mean, cmap, norm_apply)
        
    cbar = plt.colorbar(ticks = ticks)
    _ = cbar.ax.set_yticklabels(y_unique)
    
    ax.set_xlabel(axises[0])
    ax.set_ylabel(axises[1])
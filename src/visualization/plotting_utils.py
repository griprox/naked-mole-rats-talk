import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """ Computes covariance's ellipse from the cov matrix """
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy = centre, width = width, height = height,
                   angle = np.degrees(theta), **kwargs)

def plot_error_bars(data, y, sizes, cmap, norm):
    """ Plots error bars of sizes si for all xi, yi """
    for x1, x2, err, yi in zip(data[:, 0], data[:, 1], sizes, y):
        color = cmap(norm(yi))
        plt.errorbar(x1, x2, xerr = err[0], yerr = err[1],  fmt = ',', color = color)


def plot_confusion_matrix(cm, y_unique, figsize = (10, 10), cmap = plt.cm.OrRd, 
                         normalize = True, clim = (0, 1)):
    """ Plots confusion matrix cm with class labels y_unique """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize = figsize)
    im = plt.imshow(cm, interpolation='nearest', cmap = cmap)
    ax.set(xticks=np.arange(len(y_unique)),
           yticks=np.arange(len(y_unique)),
           xticklabels=y_unique, yticklabels = y_unique,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

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
    plt.colorbar(im)
    plt.clim(clim)
    return fig
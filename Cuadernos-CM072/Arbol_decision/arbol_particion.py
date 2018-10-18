import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.colors import  colorConverter, ListedColormap
cm2 = ListedColormap(['#0000aa', '#ff2020'])
from dispersion_discreta import dibuja_dispersion_discreta

def dibuja_arbol_particion(X, y, arbol, ax=None):
    if ax is None:
        ax = plt.gca()
    eps = X.std() / 2.

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]

    Z = arbol.predict(X_grid)
    Z = Z.reshape(X1.shape)
    caras = arbol.apply(X_grid)
    caras = caras.reshape(X1.shape)
    bordes = ndimage.laplace(caras) != 0
    ax.contourf(X1, X2, Z, alpha=.4, cmap=cm2, levels=[0, .5, 1])
    ax.scatter(X1[bordes], X2[bordes], marker='.', s=1)

    dibuja_dispersion_discreta(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    return ax

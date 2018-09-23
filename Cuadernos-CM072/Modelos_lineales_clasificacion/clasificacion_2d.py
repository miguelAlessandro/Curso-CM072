import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import numpy as np
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])

def dibuja_clasificacion_2d(clasificador, X, relleno=False, ax=None, eps=None, alfa=1, cm=cm3):
    # multiclase
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    valores_decision = clasificador.predict(X_grid)
    ax.imshow(valores_decision.reshape(X1.shape), extent=(x_min, x_max,
                        y_min, y_max), aspect='auto', origin='lower', alpha=alfa, cmap=cm)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
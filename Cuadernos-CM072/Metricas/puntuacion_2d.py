import matplotlib.pyplot as plt
import numpy as np

def dibuja_2d_puntuaciones(clasificador, X, ax=None, eps=None, alfa=1, cm="viridis", funcion=None):
    if eps is None:
        eps = X.std() / 2.

    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    if funcion is None:
        funcion = getattr(clasificador,"decision_function",
                           getattr(clasificador, "predict_proba"))
    else:
        funcion = getattr(clasificador, funcion)
    valores_decision = funcion(X_grid)
    if valores_decision.ndim > 1 and valores_decision.shape[1] > 1:
        # predict_proba
        valores_decision = valores_decision[:, 1]
    grr = ax.imshow(valores_decision.reshape(X1.shape), extent=(x_min, x_max, y_min, y_max), aspect='auto',
                    origin='lower', alpha=alfa, cmap=cm)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    return grr
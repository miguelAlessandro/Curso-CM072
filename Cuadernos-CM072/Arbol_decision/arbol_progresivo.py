import imageio
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
from arbol_imagen import imagen_arbol
from arbol_dibujo import dibuja_arbol

def dibuja_arbol_progresivo():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    axes = []
    for i in range(3):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                               subplot_kw={'xticks': (), 'yticks': ()})
        axes.append(ax)
    axes = np.array(axes)
    
    for i, profundidad in enumerate([1, 2, 9]):
        arbol = dibuja_arbol(X, y, profundidad=profundidad, ax=axes[i, 0])
        axes[i, 1].imshow(imagen_arbol(arbol))
        axes[i, 1].set_axis_off()

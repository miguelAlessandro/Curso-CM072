import matplotlib.pyplot as plt
import numpy as np
def dibujo_ilustracion_matriz_confusion():
    plt.figure(figsize=(8, 8))
    matriz_confusion = np.array([[401, 2], [8, 39]])
    plt.text(0.40, .7, matriz_confusion[0, 0], size=70, horizontalalignment='right')
    plt.text(0.40, .2, matriz_confusion[1, 0], size=70, horizontalalignment='right')
    plt.text(.90, .7, matriz_confusion[0, 1], size=70, horizontalalignment='right')
    plt.text(.90, 0.2, matriz_confusion[1, 1], size=70, horizontalalignment='right')
    plt.xticks([.25, .75], ["predecido 'no nueve'", "predecido 'nueve'"], size=20)
    plt.yticks([.25, .75], ["verdad 'nueve'", "verdad 'no nueve'"], size=20)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
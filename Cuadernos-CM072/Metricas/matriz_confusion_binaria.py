import matplotlib.pyplot as plt
import numpy as np

def dibuja_matriz_confusion_binaria():
    plt.text(0.45, .6, "TN", size=100, horizontalalignment='right')
    plt.text(0.45, .1, "FN", size=100, horizontalalignment='right')
    plt.text(.95, .6, "FP", size=100, horizontalalignment='right')
    plt.text(.95, 0.1, "TP", size=100, horizontalalignment='right')
    plt.xticks([.25, .75], ["prediccion negativa", "prediccion positiva"], size=15)
    plt.yticks([.25, .75], ["clase positiva", "clase negativa"], size=15)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from arbol_particion import dibuja_arbol_particion

def dibuja_arbol(X, y, profundidad=1, ax=None):
    arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=0).fit(X, y)
    ax = dibuja_arbol_particion(X, y, arbol, ax=ax)
    ax.set_title("Profundidad = %d" %profundidad) 
    return arbol

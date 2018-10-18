from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from dispersion_discreta import dibuja_dispersion_discreta
from separador_2d import dibuja_separador_2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz

def dibuja_arbol_no_monotono():
    import graphviz
    # conjunto de datos 
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    plt.figure()
    dibuja_dispersion_discreta(X[:, 0], X[:, 1], y)
    plt.legend(["Clase 0", "Clase 1"], loc="best")

    # Modelo de arboles de decision
    arbol_decision1 = DecisionTreeClassifier(random_state=0).fit(X, y)
    dibuja_separador_2d(arbol_decision1, X, estilolinea="dashed")

    # visualizamos el arbol
    export_graphviz(arbol_decision1, out_file="arbol_decision1.dot", impurity=False, filled=True)
    with open("arbol_decision1.dot") as f:
        grafo_punto1 = f.read()
    print("Caracteristicas importantes: %s" % arbol_decision1.feature_importances_)
    return graphviz.Source(grafo_punto1)

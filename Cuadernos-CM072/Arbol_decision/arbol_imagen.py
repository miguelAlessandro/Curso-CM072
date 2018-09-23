import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
from scipy import ndimage
import re

def imagen_arbol(arbol, buscar=None):
    try:
        import graphviz
    except ImportError:
        x = np.ones((10, 10))
        x[0, 0] = 0
        return x
    dato_punto = StringIO()
    export_graphviz(arbol, out_file=dato_punto, max_depth=3, impurity=False)
    dato = dato_punto.getvalue()
    dato = re.sub(r"muestras = [0-9]+\\n", "", dato)
    dato = re.sub(r"\\nmuestras = [0-9]+", "", dato)
    dato = re.sub(r"valor", "conteo", dato)

    grafo= graphviz.Source(dato, format="png")
    if buscar is None:
        buscar = "arbol"
    grafo.render(buscar)
    return imageio.imread(buscar + ".png")

import numbers
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_
import numpy as np


def hacer_blobs(n_muestras=100, n_caracteristica=2, centros=2, cluster_std=1.0,
centro_caja=(-10.0, 10.0), shuffle=True, random_state=None):
    
    generador = check_random_state(random_state)

    if isinstance(centros, numbers.Integral):
        centros = generador.uniform(centro_caja[0], centro_caja[1],
                                    size=(centros, n_caracteristica))
    else:
        centros = check_array(centros)
        n_caracteristica = centros.shape[1]

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centros)) * cluster_std

    X = []
    y = []

    n_centros = centros.shape[0]
    if isinstance(n_muestras, numbers.Integral):
        n_muestras_por_centro = [int(n_muestras // n_centros)] * n_centros
        for i in range(n_muestras % n_centros):
            n_muestras_por_centro[i] += 1
    else:
        n_muestras_por_centro = n_muestras

    for i, (n, std) in enumerate(zip(n_muestras_por_centro, cluster_std)):
        X.append(centros[i] + generador.normal(scale=std,
                                               size=(n, n_caracteristica)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        X, y = shuffle_(X, y, random_state=generador)

    return X, y
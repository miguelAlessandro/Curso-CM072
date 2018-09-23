import matplotlib.pyplot as plt
import numpy as np
from datos_desbalanceados import hacer_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dispersion_discreta import dibuja_dispersion_discreta
from separador_2d import dibuja_separador_2d
from puntuacion_2d import dibuja_2d_puntuaciones
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cm2 = ListedColormap(['#0000aa', '#ff2020'])

cdict = {'red': [(0.0, 0.0, cm2(0)[0]),
                 (1.0, cm2(1)[0], 1.0)],

         'green': [(0.0, 0.0, cm2(0)[1]),
                   (1.0, cm2(1)[1], 1.0)],

         'blue': [(0.0, 0.0, cm2(0)[2]),
                    (1.0, cm2(1)[2], 1.0)]}
ReBl = LinearSegmentedColormap("ReBl", cdict)




def dibuja_umbral_decision():

    X, y = hacer_blobs(n_muestras=(400, 50), centros=2, cluster_std=[7.0, 2],
                      random_state=22)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, random_state=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("Umbral de decision")
    axes[0, 0].set_title("datos de entrenamiento")
    dibuja_dispersion_discreta(X_entrenamiento[:, 0], X_entrenamiento[:, 1], y_entrenamiento, ax=axes[0, 0])

    svc = SVC(gamma=.05).fit(X_entrenamiento, y_entrenamiento)
    axes[0, 1].set_title("decision con  umbral 0")
    dibuja_dispersion_discreta(X_entrenamiento[:, 0], X_entrenamiento[:, 1], y_entrenamiento, ax=axes[0, 1])
    dibuja_2d_puntuaciones(svc, X_entrenamiento, funcion="decision_function", alfa=.7,
                   ax=axes[0, 1], cm=ReBl)
    dibuja_separador_2d(svc, X_entrenamiento, ancholinea=3, ax=axes[0, 1])
    axes[0, 2].set_title("decision con umbral -0.8")
    dibuja_dispersion_discreta(X_entrenamiento[:, 0], X_entrenamiento[:, 1], y_entrenamiento, ax=axes[0, 2])
    dibuja_separador_2d(svc, X_entrenamiento, ancholinea=3, ax=axes[0, 2], umbral=-.8)
    dibuja_2d_puntuaciones(svc, X_entrenamiento, funcion="decision_function", alfa=.7,
                   ax=axes[0, 2], cm=ReBl)

    axes[1, 0].set_axis_off()

    mascara_bool= np.abs(X_entrenamiento[:, 1] - 7) < 5
    bla = np.sum(mascara_bool)

    linea = np.linspace(X_entrenamiento.min(), X_entrenamiento.max(), 100)
    axes[1, 1].set_title("Seccion cruzada con umbral 0")
    axes[1, 1].plot(linea, svc.decision_function(np.c_[linea, 10 * np.ones(100)]), c='k')
    dec = svc.decision_function(np.c_[linea, 10 * np.ones(100)])
    contorno = (dec > 0).reshape(1, -1).repeat(10, axis=0)
    axes[1, 1].contourf(linea, np.linspace(-1.5, 1.5, 10), contorno, alpha=0.4, cmap=cm2)
    dibuja_dispersion_discreta(X_entrenamiento[mascara_bool, 0], np.zeros(bla), y_entrenamiento[mascara_bool], 
                               ax=axes[1, 1])
    axes[1, 1].set_xlim(X_entrenamiento.min(), X_entrenamiento.max())
    axes[1, 1].set_ylim(-1.5, 1.5)
    axes[1, 1].set_xticks(())
    axes[1, 1].set_ylabel("Valor de decision")

    contorno2 = (dec > -.8).reshape(1, -1).repeat(10, axis=0)
    axes[1, 2].set_title("Seccion cruzada con umbral  -0.8")
    axes[1, 2].contourf(linea, np.linspace(-1.5, 1.5, 10), contorno2, alpha=0.4, cmap=cm2)
    dibuja_dispersion_discreta(X_entrenamiento[mascara_bool, 0], np.zeros(bla), y_entrenamiento[mascara_bool], 
                               alfa=.1, ax=axes[1, 2])
    axes[1, 2].plot(linea, svc.decision_function(np.c_[linea, 10 * np.ones(100)]), c='k')
    axes[1, 2].set_xlim(X_entrenamiento.min(), X_entrenamiento.max())
    axes[1, 2].set_ylim(-1.5, 1.5)
    axes[1, 2].set_xticks(())
    axes[1, 2].set_ylabel("Valor de decision")
    axes[1, 0].legend(['Clase negativa', 'Clase positiva'])
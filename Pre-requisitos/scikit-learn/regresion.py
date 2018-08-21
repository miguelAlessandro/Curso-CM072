import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_lineal_regresion():
    a = 0.5
    b = 1.0

    # x desde 0 a 10
    x = 30 * np.random.random(20)

    # y = a*x + b 
    y = a * x + b + np.random.normal(size=x.shape)

    # creamos un clasificador de regresion  lineal
    clf = LinearRegression()
    clf.fit(x[:, None], y)

    # predictor  de y para la data
    x_new = np.linspace(0, 30, 100)
    y_new = clf.predict(x_new[:, None])

    # Graficamos  los  resultados
    ax = plt.axes()
    ax.scatter(x, y)
    ax.plot(x_new, y_new)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.axis('tight')


if __name__ == '__main__':
    plot_lineal_regresion()
    plt.show()

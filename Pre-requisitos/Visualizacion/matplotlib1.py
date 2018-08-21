# Visualizacion de datos: Matplotlib

import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def grafico_lineas():
    fechas = [1944, 1954, 1983, 1986, 1988, 1992, 1997]
    gdp =[300.2, 567.3, 1075.3, 2789.4, 5965.5, 10345.5, 14897.2]

 # Creamos un gráfico de linea, con fechas en el eje X y gdp en el eje Y
    plt.plot(fechas, gdp, color='red', marker='o', linestyle ='solid')

 # Agregamos un título
    plt.title("Grafico de Linea")

 # Agregamos una etiqueta al eje Y 
    plt.ylabel("Ganancias en $")
    plt.show()
    
def grafico_barras():
    peliculas = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
    num_oscars = [5, 11, 3, 8, 10]

 # Centremos las barras
    xs = [i + 0.1 for i, _ in enumerate(peliculas)]

    plt.bar(xs, num_oscars)

    plt.ylabel("# de Oscars")
    plt.title("Peliculas Galardonadas con el Oscar")

 # Etiquetamos el eje X con el nombre de las peliculas en el centro de las barras

    plt.xticks([i + 0.5 for i, _ in enumerate(peliculas)], peliculas)

    plt.show()
    
    
def grafico_lineas2():
    varianza= [1,2,4,8,16,32,64,128,256]
    bs_2= [256,128,64,32,16,8,4,2,1]
    total_error = [x + y for x, y in zip(varianza, bs_2)]
    xs = range(len(varianza))

 # Podemos llamar a plt.plot varias veces para mostrar
 # un grupo de figuras del mismo 'grafico'

    plt.plot(xs, varianza,     'g-',  label='varianza')    
    plt.plot(xs, bs_2, 'r-.', label='bias^2')      
    plt.plot(xs, total_error,  'b:',  label='total error') 

 # Podemos asignas lebels s ese grupo de figuras
 # Contamos con una leyenda para el grafico resultante
 # loc=9 significa "top center"
    
    plt.legend(loc=9)
    plt.xlabel("Modelo de  complexidad")
    plt.title("Bias-Varianza")
    plt.show()
    
def grafico_dispersion():
    fr = [45, 78, 67, 89, 71, 68, 60, 74, 65]
    minu = [145, 167, 132, 156, 143, 122, 114, 183, 143]
    labels = ['a', 'b', 'c', 'd','e','f','g', 'h','i']
    
    plt.scatter(fr, minu)

# Etiquetamos cada punto
    for label, f_c, m_c in zip(labels, fr, minu):
        plt.annotate(label,
            xy = (f_c, m_c),
            xytext = (-5, 5),
            textcoords='offset points')
    
    
    plt.title("Minutos diarios-Numero de amigos")
    plt.xlabel("Numero de amigos")
    plt.ylabel("Minutos diarios en el sitio ")
    plt.show()

def ejemplo_histograma():
    mu = 100  # media de la distribución
    sigma = 15   # desviación estándar de la distribución
    x = mu + sigma * np.random.randn(10000)
    
    num_bins = 50
    # El histograma de la data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)

    # agregamos una mejor 'linea de ajuste ' 
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Habilidades')
    plt.ylabel('Probabilidad')
    plt.title(r'Histograma de IQ: $\mu=100$, $\sigma=15$')
    
    # Ajustando el espaciado en la figura
    plt.subplots_adjust(left=0.15)
    plt.show()
    
if __name__ == "__main__":
    
    grafico_lineas()
    grafico_barras()
    grafico_lineas2()
    grafico_dispersion()
    ejemplo_histograma()



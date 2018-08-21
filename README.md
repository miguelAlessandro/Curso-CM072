# Curso CM-072 Una introducción al Machine Learning con Python

# Temario 

## Tarea 1

## Lecturaa adicionales

###  Clases 1-2

*  Artículo de Jake VanderPlas sobre la velocidad de Python [Why Python is Slow: Looking Under the Hood](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/).
* Artículo de O'really acerca de que es la ciencia de datos y las proyecciones a futuro  [What is data science? The future belongs to the companies and people that turn data into products.](https://www.oreilly.com/ideas/what-is-data-science).
* [The End of Theory: The Data Deluge Makes the Scientific Method Obsolete](https://www.wired.com/2008/06/pb-theory/), presenta un análisis acerca de las deficiencia del método científico actual.
* [Analyzing the Analyzers ](http://cdn.oreillystatic.com/oreilly/radarreport/0636920029014/Analyzing_the_Analyzers.pdf) proporciona una mirada útil a los diferentes tipos de científicos de datos.

## Herramientas a  usar 

### Anaconda 

[Anaconda](https://www.continuum.io/downloads) es una distribución completa  libre de [Python](https://www.python.org/) incluye [paquetes de Python ](http://docs.continuum.io/anaconda/pkg-docs).

Anaconda incluye los instaladores de Python 2.7 y 3.5.  La instalación en **Linux**, se encuentra en la página de Anaconda y es 
más o menos así

1 . Descargar el instalador de Anaconda para Linux.

2 . Después de descargar el instalar, en el terminal, ejecuta para 3.5

```bash
c-lara@Lara:~$ bash Anaconda3-2.4.1-Linux-x86_64.sh

```

Es recomendable leer, alguna de las característica de Anaconda en el siguiente material [conda 30-minutes test drive](http://conda.pydata.org/docs/test-drive.html).

3 . La instalación de paquetes como [seaborn](http://stanford.edu/~mwaskom/software/seaborn/) o [bokeh](http://bokeh.pydata.org/en/latest/) se pueden realizar a través de Anaconda, de la siguiente manera:



``` bash
c-lara@Lara:~$ conda install bokeh
```
Alternativamente podemos desde PyPI usando **pip**:

```bash
c-lara@Lara:~$ pip install bokeh
``` 

El proyecto [Anaconda](https://www.continuum.io/downloads) ha creado [R Essentials](http://anaconda.org/r/r-essentials), que incluye el IRKernel y alrededor de 80 paquetes para análisis de datos, incluyendo `dplyr`, `shiny`, `ggplot2`,`caret`, etc. Para instalar **R Essentials** en un entorno de trabajo, hacemos

```bash
c-lara@Lara:~$ conda install -c r r-essentials
``` 

### Proyecto Jupyter y el Jupyter Nbviewer

El [Proyecto Jupyter](http://jupyter.org/)  es una aplicación web que te permite crear y compartir documentos que contienen código de diversos lenguajes de programación, ecuaciones,  visualizaciones y texto en diversos formatos. El uso de Jupyter incluye la ciencia de datos, simulación numérica, la modelización en estadística, Machine Learning, etc.


[Jupyter nbviewer](https://nbviewer.jupyter.org/)  es un servicio web gratuito que te permite compartir las versiones de archivos realizados por Jupyter, permitiendo el renderizado de diversos fórmatos incluyendo, código latex.

- [Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/).

[Unofficial Jupyter Notebook Extensions](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/) contiene una colección de extensiones no oficiales de la comunidad que añaden funcionalidad a Jupyter notebook. Estas extensiones están escritas principalmente en Javascript y se cargarán localmente en su navegador.

```bash
c-lara@Lara:~$ pip install jupyter_contrib_nbextensions
``` 

O utilizando conda

```bash
c-lara@Lara:~$ conda install -c conda-forge jupyter_contrib_nbextensions
``` 

## Software

* [NumPy](http://www.numpy.org/), es la biblioteca natural para  python numérico. La característica más potente de NumPy es la  matriz n-dimensional. Esta biblioteca  contiene funciones básicas de álgebra lineal, transformadas de Fourier, capacidades avanzadas de números aleatorios y herramientas para la integración con otros lenguajes de bajo nivel como Fortran, C y C ++.

* [SciPy](https://www.scipy.org/) es la biblioeteca para python científico. SciPy se basa en NumPy y es una de las bibliotecas más útiles por la variedad de módulos de ciencia y ingeniería de alto nivel con la que cuenta, como la transformada discreta de Fourier,  álgebra lineal, optimización,  matrices dispersas, etc.

* [Matplotlib](http://matplotlib.org/) es una librería de Python  para  crear una gran variedad de gráficos, a partir de histogramas, lineas, etc, usando si es necesario  comandos de látex para agregar matemáticas a los gráficos.

* [Pandas](http://pandas.pydata.org/) es una librería  para operaciones y manipulaciones de datos estructurados. Pandas ha sido añadido  recientemente a Python y han sido fundamental para impulsar el uso de Python en la ciencia de datos.

* [Scikit-learn](http://scikit-learn.org/stable/), es tal vez la mejor biblioteca para Machine Learning, construida sobre NumPy, SciPy y Matplotlib, esta biblioteca contiene una gran cantidad de herramientas eficientes para el Machine Learning y el modelado estadístico incluyendo clasificación, regresión, agrupación y reducción de la dimensionalidad.

* [Seaborn](https://seaborn.pydata.org/) es una libreria para la visualización de datos estadísticos. Seaborn es una biblioteca para hacer atractivos e informativos los gráficos estadísticos en Python. Se basa en matplotlib. Seaborn pretende hacer de la visualización una parte central de la exploración y la comprensión de los datos.


### Git y Github

[Git](https://git-scm.com/) es un sistema de control de versiones de gran potencia y versatilidad en el manejo de un gran número de archivos de  código fuente a a través del desarrollo no lineal, es decir vía la gestión rápida de ramas y mezclado de Para poder revisar y aprender los comandos necesarios de Git, puedes darle una ojeada al excelente [tutorial de CodeSchool](https://try.github.io/levels/1/challenges/1) o a la [guía](http://rogerdudler.github.io/git-guide/index.es.html) de Roger Dudle para aprender  Git.

[Github](https://github.com/) es una plataforma de desarrollo colaborativo de software utilizado para alojar proyectos (muchos proyectos importantes como paquetes de R, Django, el Kernel de Linux, se encuentran alojados ahí) utilizando Git y el framework Ruby on Rails.

Podemos instalar Git en Ubuntu utilizando el administrador de paquetes `Apt`:

```bash
c-lara@Lara:~$sudo apt-get update
c-lara@Lara:~$sudo apt-get install git
```

Referencias y Lecturas:

- [Usando el GIT](http://www.cs.swarthmore.edu/~newhall/unixhelp/git.php).
- [Practical Git Introduction](http://marc.helbling.fr/2014/09/practical-git-introduction).
- [Visual Git Guide](http://marklodato.github.io/visual-git-guide/index-es.html).

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
#import warnings

#warnings.filterwarnings("ignore")


# Grafico de la funcion seno y coseno en seaborn
def fun_sin_cos():
    sns.set()
    x = np.linspace(0, 10, 1000)
    plt.plot(x, np.sin(x), x, np.cos(x))
    plt.show()

# Grafico de un histograma 
def histograma():   
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y'])

    for col in 'xy':
        plt.hist(data[col], normed=True, alpha=0.5)
    
    plt.show()   

# Podemos conseguir un buen estimado de la distribucion, usando
# el KDE (Kernel density estimation)

def seaborn_kde():   
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y'])

    for col in 'xy': 
        sns.kdeplot(data[col], shade=True) 
    plt.show()    

# Los histogramas y el KDE pueden ser  combinados usando displot
    
def seaborn_displot():   
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y']) 
    sns.distplot(data['x'])
    
    plt.show()

# Podemos conseguir visualizacion en dos dimensiones, con kdeplot 
def seaborn_kde2():
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y']) 
    for col in 'xy': 
        sns.kdeplot(data) 
    plt.show()
     
 # Podemos observar las distribuciones marginales y conjuntas (joint)
 # con jointplot
 
def seaborn_join():
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x', 'y']) 
    with sns.axes_style('white'):
        sns.jointplot("x", "y", data, kind='hex')
    
    plt.show()
     
    
    
if __name__ == "__main__":
    fun_sin_cos()
    histograma()
    seaborn_kde()
    seaborn_displot()
    seaborn_kde2()
    seaborn_join()

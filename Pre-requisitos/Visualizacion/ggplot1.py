# Uso de la libreria ggplot. Mayor información:
# http://ggplot.yhathq.com/docs/index.html

from matplotlib import pyplot as plt
import scipy
import numpy as np
import pandas as pd
from ggplot import *

def ejemplo_ggplot():
    p = ggplot(mtcars, aes('cyl'))
    print (p + geom_bar())

plt.show(1)

def dnorm(x, mean, var):
    return scipy.stats.norm(mean,var).pdf(x)
    
data = pd.DataFrame({'x':np.arange(-5,6)})
ggplot(aes(x='x'),data=data) + \
    stat_function(fun=dnorm,color="blue",
                  args={'mean':0.0,'var':0.2}) + \
    stat_function(fun=dnorm,color="red",
                  args={'mean':0.0,'var':1.0}) + \
    stat_function(fun=dnorm,color="yellow",
                  args={'mean':0.0,'var':5.0}) + \
    stat_function(fun=dnorm,color="green",
                  args={'mean':-2.0,'var':0.5})
plt.show()

if __name__ == "__main__":
    ejemplo_ggplot()

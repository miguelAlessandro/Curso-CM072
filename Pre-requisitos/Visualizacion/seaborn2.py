import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

sns.set()

# Pairplot

def pairplot():
    iris = sns.load_dataset("iris")
    iris.head()
    sns.pairplot(iris, hue="species", size=2.5)
    
    plt.show()

# Uso de facetgrid en seaborn
def facet_grid():

    tips = sns.load_dataset('tips')
    tips.head()
    tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

    grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
    grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));
    plt.show()
    
 # Podemos graficar series de tiempo con seaborn
 
def series_tiempo():
    
    planets = sns.load_dataset('planets')
    planets.head()
    
    with sns.axes_style('white'):
        g = sns.factorplot("year", data=planets, aspect=1.5)
        g.set_xticklabels(step=5)
        
    plt.show()




if __name__ == "__main__":
    pairplot()
    facet_grid()
    series_tiempo()

# Ejemplo tomado de la documentacion de plotly
# https://plot.ly/python/#3d-charts


# El procemiento de ejecución del programa es

# Ejecutamos en el terminal de python, despues de habernos
# inscrito en la pagina de plotly 
#https://plot.ly 

# python -c "import plotly; plotly.tools.set_credentials_file(username='xxxxx', api_key='xxxxxxxx')"
# username y api_key se generan cuando te inscribes en la pagina

# Y despues, correr el script

# python plotly1.py


import plotly.plotly as py
from plotly.graph_objs import *
#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

init_notebook_mode()

def dist_planetas():
    
# Agregamos datos 
    planetas = ['Mercurio', 'Venus', 'Tierra', 'Marte', 'Jupiter', 'Saturno', 'Urano', 'Neptuno', 'Pluton']
    color_planeta = ['rgb(135, 135, 125)', 'rgb(210, 50, 0)', 'rgb(50, 90, 255)',
                 'rgb(178, 0, 0)', 'rgb(235, 235, 210)', 'rgb(235, 205, 130)',
                 'rgb(55, 255, 217)', 'rgb(38, 0, 171)', 'rgb(255, 255, 255)']
    distancia_al_sol = [57.9, 108.2, 149.6, 227.9, 778.6, 1433.5, 2872.5, 4495.1, 5906.4]
    densidad = [5427, 5243, 5514, 3933, 1326, 687, 1271, 1638, 2095]
    gravedad = [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
    diametro_planeta = [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370]

# Creando una traza y el grafico de los planetas de acuerdo a su diametro
    trace1 = Scatter3d(
        x = distancia_al_sol,
        y = densidad,
        z = gravedad,
        text = planetas,
        mode = 'markers',
        marker = dict(
            sizemode = 'diametro',
            sizeref = 750, # informacion: https://plot.ly/python/reference/#scatter-marker-sizeref
            size = diametro_planeta,
            color = color_planeta,
        )  
)
    data=[trace1]

# Estilo 
    layout=Layout(width=800, height=800, title = 'planetas!',
                  scene = dict(xaxis=dict(title='Distancia al sol',
                                          titlefont=dict(color='Orange')),
                                yaxis=dict(title='densidad',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                                zaxis=dict(title='gravedad',
                                       titlefont=dict(color='rgb(220, 220, 220)')),
                                bgcolor = 'rgb(20, 24, 54)'
                              )
             )

# Plot

    fig=dict(data=data, layout=layout)
    py.iplot(fig, filename='El_tamaño_de_los_planetas_sistema_solar')
    
if __name__ == "__main__":
    dist_planetas()

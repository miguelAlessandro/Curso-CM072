import pygal

# http://blog.codeeval.com/codeevalblog/2014

data = [
  ["Python", 30.3],
  ["Java", 22.2],
  ["C++", 13],
  ["Ruby", 10.6],
  ["Javascript", 5.2],
  ["C#", 5],
  ["C", 4.1],
  ["PHP", 3.3],
  ["Perl", 1.6],
  ["Go", 1.5],
  ["Haskell", 1.2],
  ["Scala", 1],
  ["Objective-C", 0.4],
  ["Clojure", 0.2],
  ["Bash", 0.1],
  ["Lua", 0.04],
  ["TCL", 0.03]
]
    

# Hacemos un gráfico circular en Pygal

pie_chart = pygal.Pie()

# Agregando un título
pie_chart.title = "CodeEval: Lenguajes de Programación Populares del 2014 "

# agregamos los datos 
for label, data_points in data:
    pie_chart.add(label, data_points)

# Enviamos el grafico a formato svg
pie_chart.render_to_file('Ranking_Programas.svg')


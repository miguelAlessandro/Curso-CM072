# -*- coding: utf-8 -*-

import numpy as np
import scipy.special

from bokeh.plotting import figure, show, output_file, vplot


#Distribucion Exponencial

p0 = figure(title="Distribucion Exponencial (μ=1)",tools="save",
            background_fill_color="#E8DDCB")

mu = 1
medido = np.random.exponential(mu, 5000)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(0, 8.0, 5000)
pdf=mu*np.exp(-mu*x)
cdf=1 -np.exp(-mu*x)

p0.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
        fill_color="#036564", line_color="#033649")
p0.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p0.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")

p0.legend.location = "top_left"
p0.xaxis.axis_label = 'x'
p0.yaxis.axis_label = 'Pr(x)'


#Distribucion Normal

p1 = figure(title="Distribucion Normal (μ=0, σ=0.5)",tools="save",
            background_fill_color="#E8DDCB")

mu, sigma = 0, 0.5

medido = np.random.normal(mu, sigma, 1000)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(-2, 2, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

p1.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
        fill_color="#036564", line_color="#033649")
p1.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p1.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")

p1.legend.location = "top_left"
p1.xaxis.axis_label = 'x'
p1.yaxis.axis_label = 'Pr(x)'

# Distribucion Log-Normal

p2 = figure(title="Distribucion Log Normal  (μ=0, σ=0.5)", tools="save",
            background_fill_color="#E8DDCB")

mu, sigma = 0, 0.5

medido = np.random.lognormal(mu, sigma, 1000)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(0, 8.0, 1000)
pdf = 1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))
cdf = (1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2

p2.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
        fill_color="#036564", line_color="#033649")
p2.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p2.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")

p2.legend.location = "bottom_right"
p2.xaxis.axis_label = 'x'
p2.yaxis.axis_label = 'Pr(x)'


# Distribucion Gamma

p3 = figure(title="Distribucion Gamma (k=1, θ=2)", tools="save",
            background_fill_color="#E8DDCB")

k, theta = 1.0, 2.0

medido = np.random.gamma(k, theta, 1000)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(0, 20.0, 1000)
pdf = x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k))
cdf = scipy.special.gammainc(k, x/theta) / scipy.special.gamma(k)

p3.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
        fill_color="#036564", line_color="#033649")
p3.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p3.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")

p3.legend.location = "top_left"
p3.xaxis.axis_label = 'x'
p3.yaxis.axis_label = 'Pr(x)'

# Distribucion beta 

p4 = figure(title="Distribucion Beta (α=2, β=2)", tools="save",
            background_fill_color="#E8DDCB")

alpha, beta = 2.0, 2.0

medido = np.random.beta(alpha, beta, 1000)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(0, 1, 1000)
pdf = x**(alpha-1) * (1-x)**(beta-1) / scipy.special.beta(alpha, beta)
cdf = scipy.special.btdtr(alpha, beta, x)

p4.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
        fill_color="#036564", line_color="#033649")
p4.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p4.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")

p4.xaxis.axis_label = 'x'
p4.yaxis.axis_label = 'Pr(x)'

# Distribucion de Weibull

p5 = figure(title="Distribution Weibull (λ=1, k=1.25)", tools="save",
            background_fill_color="#E8DDCB")

lam, k = 1, 1.25

medido = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)
hist, bordes = np.histogram(medido, density=True, bins=50)

x = np.linspace(0, 8, 1000)
pdf = (k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k)
cdf = 1 - np.exp(-(x/lam)**k)

p5.quad(top=hist, bottom=0, left=bordes[:-1], right=bordes[1:],
       fill_color="#036564", line_color="#033649")
p5.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
p5.line(x, cdf, line_color="blue", line_width=2, alpha=0.7, legend="CDF")


p5.legend.location = "top_left"
p5.xaxis.axis_label = 'x'
p5.yaxis.axis_label = 'Pr(x)'


output_file('histograma.html', title="histogramas por ejemplos ")

show(vplot(p0,p1,p2,p3,p4,p5))

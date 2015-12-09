# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from math import pi, exp
import numpy as np
from textwrap import wrap
import matplotlib

def f(x):
	return 1.0/(1.0+exp(-4*(x-0)))

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(-1,1,400)
T = np.array(list(map(f, X)))
ax.plot(X,T,'b', label=u'função logística')
X = np.linspace(-1,1,50)
noise = np.random.normal(0.0,0.1,len(X))
T = np.array(list(map(f, X))) + noise
ax.scatter(X,T,s=80,c='green',marker='+', label=u'dados de treinamento')

title = ax.set_title("\n".join(wrap(u'Base de treinamento f(x) com ruído gaussiano', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend()
fig.savefig("f_4v1_train_data_paper.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X,T,s=80,c='green',marker='+', label=u'dados de treinamento')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(4,1)
r.train(train, 0.0, 10000, 0.1,1.0)

X = np.linspace(-1,1,400)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função logística com 4 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend()
fig.savefig("f_4v1_model_paper.png")
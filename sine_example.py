# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from math import sin, pi
import numpy as np
from textwrap import wrap

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.arange(0,4*pi,0.2)
T = np.sin(X)
ax.scatter(X,T,s=80,c='green',marker='+', label=u'dados de treinamento')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(45,1)
r.train(train, 0.0, 1000000, 0.1, 1.0)

X = np.linspace(0,4*pi,800)
T = np.sin(X)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,T,'b', label=u'função seno')
ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função seno com 45 neurônios RBF e variância 1.0 (1milhão épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('sin(x)')
ax.legend()
plt.savefig("train_sin45n_v1.0.png")

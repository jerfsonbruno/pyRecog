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

X2 = np.linspace(0,2*pi,400)
T2 = np.cos(X2)
ax.plot(X2,T2,'b', label=u'função cosseno')
X3 = np.linspace(0,2*pi,50)
noise = np.random.normal(0.0,0.1,len(X3))
T3 = np.cos(X3) + noise
ax.scatter(X3,T3,s=80,c='green',marker='+', label=u'dados de treinamento')

title = ax.set_title("\n".join(wrap(u'Base de treinamento cos(x) com ruído gaussiano', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('cos(x)')
ax.legend(bbox_to_anchor=[0.8, 1])
fig.savefig("cos_5v1_train_data_paper.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X3,T3,s=80,c='green',marker='+', label=u'dados de treinamento')
X4 = np.asmatrix(X3)
T4 = np.asmatrix(T3)
t_set = np.concatenate((X4,T4), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(5,1)
r.train(train, 0.0, 10000, 0.1,1.0)

Y = []
for i in X2:
	Y += [r([i])[0]]

ax.plot(X2,Y,'r', label=u'RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função cosseno com 5 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('cos(x)')
ax.legend(bbox_to_anchor=[0.8, 1])
fig.savefig("cos_5v1_model_paper.png")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(X3,T3,s=80,c='green',marker='+', label=u'dados de treinamento')
ax.plot(X2,T2,'b', label=u'função cosseno')
ax.plot(X2,Y,'r', label=u'RBF')
title = ax.set_title("\n".join(wrap(u'Aproximando função cosseno com 5 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('cos(x)')
ax.legend(bbox_to_anchor=[0.8, 1])
fig.savefig("cos_5v1_all_paper.png")

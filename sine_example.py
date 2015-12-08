# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from math import sin, pi
import numpy as np
from textwrap import wrap

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(0,2*pi,400)
T = np.cos(X)
ax.plot(X,T,'b', label=u'função seno')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(3,1)
r.train(train, 0.0, 10000, 0.1)

X = np.linspace(0,2*pi,400)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função seno com 3 neurônios RBF e variância automática (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('sin(x)')
ax.legend()
fig.savefig("3_va.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(0,2*pi,400)
T = np.cos(X)
ax.plot(X,T,'b', label=u'função seno')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(6,1)
r.train(train, 0.0, 10000, 0.1)

X = np.linspace(0,2*pi,400)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função seno com 6 neurônios RBF e variância automática (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('sin(x)')
ax.legend()
plt.savefig("6_va.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(0,2*pi,400)
T = np.cos(X)
ax.plot(X,T,'b', label=u'função seno')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(3,1)
r.train(train, 0.0, 10000, 0.1, 1.0)

X = np.linspace(0,2*pi,400)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função seno com 3 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('sin(x)')
ax.legend()
plt.savefig("3_v1.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(0,2*pi,400)
T = np.cos(X)
ax.plot(X,T,'b', label=u'função seno')
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(6,1)
r.train(train, 0.0, 10000, 0.1, 1.0)

X = np.linspace(0,2*pi,400)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label='RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função seno com 6 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('sin(x)')
ax.legend()
plt.savefig("6_v1.png")

plt.figure()
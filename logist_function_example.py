import matplotlib.pyplot as plt
import numpy as np
from math import exp
from textwrap import wrap

def f(x):
	return 1.0/(1.0+exp(-4*(x-0)))

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(-1,1,20)
T = np.array(list(map(f, X)))
ax.plot(X,T,'b', label=u'função logística')
#plt.show()
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(3,1)
r.train(train, 0.0, 10000, 0.1, 0.5)

X = np.linspace(-1,1,20)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label=u'RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função logística com 3 neurônios RBF e variância 0.5 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend(bbox_to_anchor=[1, 0.2])
plt.savefig("lf3n_v.5.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(-1,1,20)
T = np.array(list(map(f, X)))
ax.plot(X,T,'b', label=u'função logística')
#plt.show()
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(6,1)
r.train(train, 0.0, 10000, 0.1, 0.5)

X = np.linspace(-1,1,20)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label=u'RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função logística com 6 neurônios RBF e variância 0.5 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend(bbox_to_anchor=[1, 0.2])
plt.savefig("lf6n_v.5.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(-1,1,20)
T = np.array(list(map(f, X)))
ax.plot(X,T,'b', label=u'função logística')
#plt.show()
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(3,1)
r.train(train, 0.0, 10000, 0.1, 1.0)

X = np.linspace(-1,1,20)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label=u'RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função logística com 3 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend(bbox_to_anchor=[1, 0.2])
plt.savefig("lf3n_v1.png")

fig = plt.figure()
ax = fig.add_subplot(111)

X = np.linspace(-1,1,20)
T = np.array(list(map(f, X)))
ax.plot(X,T,'b', label=u'função logística')
#plt.show()
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(6,1)
r.train(train, 0.0, 10000, 0.1, 1.0)

X = np.linspace(-1,1,20)
Y = []
for i in X:
	Y += [r([i])[0]]

ax.plot(X,Y,'r', label=u'RBF')

title = ax.set_title("\n".join(wrap(u'Aproximando função logística com 6 neurônios RBF e variância 1.0 (10mil épocas)', 60)))
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.xlabel('x')
plt.ylabel('f(x)')
ax.legend(bbox_to_anchor=[1, 0.2])
plt.savefig("lf6n_v1.png")
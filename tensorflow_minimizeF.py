#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf;


minimum=[1,3]

#define some function using thaeano Ops (e.g. T.dot for matrix multiplication)
# this function is f(w)=||w-minimum||^2, and so has minimum at minimum, i.e. at vector [1,3]
def f(w):
    shiftedW=w-np.array(minimum);
    return tf.reduce_sum(tf.multiply(shiftedW,shiftedW));

#define starting value of W for gradient descent
#here, W is a 2D vector
initialW=np.random.rand(2)

#create a shared variable (i.e. a variable that persists between calls to a theano function)
w = tf.Variable(initialW,name="w");

#define output of applying f to w
#out goal will be to minimize f(w), i.e. find w with lowest possible f(w)
z=f(w);

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(z)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

with sess:
    for i in range(100):
      train.run();
      npw=w.eval();
#      npw=sess.run(w)
      print(npw)
#sess.close()

print("True minimum: "+str(minimum));
print("Found minimum:"+str(npw));

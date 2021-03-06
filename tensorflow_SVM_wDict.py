#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

"""

import numpy as np
import tensorflow as tf
import sys

np.random.seed(123)

args = sys.argv

if len(args) != 2:
    print "Correct usage: python tensorflow_SVM_wDict.py <beta>"
    sys.exit(0)

beta = float(args[1])

numAttributes = 10

def makeDataset():
    sCnt = 1000

    # true parameters w and b
    # seperate the weights to num attributes
    true_w1 = 0.3
    true_w2 = -0.6
    true_w3 = -0.1
    true_w4 = -0.5
    true_w5 = 0.6
    true_w6 = -0.4
    true_w7 = -0.2
    true_w8 = 0.35
    true_w9 = 0.5
    true_w10 = 0.55
    true_b = 0.7

    # sample some random point in 2D feature space
    x_train = np.random.randn(sCnt, numAttributes).astype(dtype='float32')

    # calculate u=w^Tx+b
    u = true_w1 * x_train[:, 0] + \
        true_w2 * x_train[:, 1] + \
        true_w3 * x_train[:, 2] + \
        true_w4 * x_train[:, 3] + \
        true_w5 * x_train[:, 4] + \
        true_w6 * x_train[:, 5] + \
        true_w7 * x_train[:, 6] + \
        true_w8 * x_train[:, 7] + \
        true_w9 * x_train[:, 8] + \
        true_w10 * x_train[:, 9] + \
        true_b

    # P(+1|x)=a(u) #see slides for def. of a(u)
    pPlusOne = 1.0 / (1.0 + np.exp(-1.0 * u))

    # sample realistic (i.e. based on pPlusOne, but not deterministic) class values for the dataset
    # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
    y01_train = np.random.binomial(1, pPlusOne)
    y_train = 2 * y01_train - 1
    y_train = y_train.reshape((sCnt, 1)).astype(dtype='float32')

    x_test = np.random.randn(sCnt, numAttributes).astype(dtype='float32')

    # calculate u=w^Tx+b
    u = true_w1 * x_test[:, 0] + \
        true_w2 * x_test[:, 1] + \
        true_w3 * x_test[:, 2] + \
        true_w4 * x_test[:, 3] + \
        true_w5 * x_test[:, 4] + \
        true_w6 * x_test[:, 5] + \
        true_w7 * x_test[:, 6] + \
        true_w8 * x_test[:, 7] + \
        true_w9 * x_test[:, 8] + \
        true_w10 * x_test[:, 9] + \
        true_b

    # P(+1|x)=a(u) #see slides for def. of a(u)
    pPlusOne = 1.0 / (1.0 + np.exp(-1.0 * u))

    # sample realistic (i.e. based on pPlusOne, but not deterministic) class values for the dataset
    # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
    y01_test = np.random.binomial(1, pPlusOne)
    y_test = 2 * y01_test - 1
    y_test = y_test.reshape((sCnt, 1)).astype(dtype='float32')

    train_adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype='float32')
    train_degree_matrix = np.multiply(np.array([1, 2, 1, 2, 1, 1, 1, 1, 1, 2], dtype='float32'),
                                      np.identity(numAttributes))

    laplacian_matrix = np.subtract(train_degree_matrix, train_adjacency_matrix)

    q_matrix = np.linalg.inv(np.add(np.identity(numAttributes), beta * laplacian_matrix))
    return x_train, y_train, x_test, y_test, q_matrix


## we use the dataset with x_train being the matrix "n by 2" with samples as rows, and the two features as columns
## y_train is the true class (-1 vs 1), we have it as a matrix "n by 1"
x_train, y_train, x_test, y_test, q_matrix = makeDataset()
n_train = x_train.shape[0]
n_test = x_test.shape[0]
fCnt = x_train.shape[1]
#### START OF LEARNING

n_epochs = 100
batch_size = 128

## define variables for tensorflow

##define and initialize shared variables
## (the variable persist, they encode the state of the classifier throughout learning via gradient descent)
# w is the feature weights, a 2D vector
initialW = np.random.rand(numAttributes, 1).astype(dtype='float32')
w = tf.Variable(initialW, name="w")

# b is the bias, so just a single number
initialB = 0.0
b = tf.Variable(initialB, name="b")

## define non-shared/placeholder variable types
# x will be our [#samples x #features] matrix of all training samples
# in this example, we'll have 2 features
x = tf.placeholder(dtype=tf.float32, name='x')
# y will be our vector of classess for all training samples
y = tf.placeholder(dtype=tf.float32, name='y')

## set up new variables that are functions/transformations of the above
# predicted class for each sample (a vector)
# tf.matmul(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
# thre result is a vector with #samples entries
predictions = tf.matmul(x, w) + b
# loss for each sample (a vector)
loss = tf.maximum(1 - tf.multiply(y, predictions), 0)
# risk over all samples (a number)
risk = tf.reduce_mean(loss)

# set the penalty for w^TQw
NICK_penalty = tf.multiply(tf.transpose(w), tf.multiply(tf.cast(q_matrix, tf.float32), w))

# add it to risk
cost = risk + NICK_penalty

# define which optimizer to use
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(cost)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_pred = sess.run([predictions], feed_dict={x: x_test, y: y_test})
acc = np.sum(np.sign(1.0 + np.multiply(y_test, np.sign(y_pred)))) / n_test
print(acc)
# start the iterations of gradient descent
for i in range(0, n_epochs):
    for j in range(np.random.randint(0, int(batch_size / 2)), n_train, batch_size):
        jS = j
        jE = min(n_train, j + batch_size)
        x_batch = x_train[jS:jE, :]
        y_batch = y_train[jS:jE, :]
        _, curr_cost, predY = sess.run([train, cost, predictions], feed_dict={x: x_batch, y: y_batch})
    y_pred = sess.run([predictions], feed_dict={x: x_test, y: y_test})
    acc = np.sum(np.sign(1.0 + np.multiply(y_test, np.sign(y_pred)))) / n_test
    print(acc)

#!/usr/bin/env python

"""
Tutorial link: https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html

# Run after activating the tensorflow environment
$ python mnist_softmax_beginner.py

"""

# Installing the MNIST data sets into "MNIST_data" folder
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("MNIST data loaded.")


# Running the softmax regression model

import tensorflow as tf
print("Starting to build the TensorFlow computation graph...")

# Setting 'x' as placeholder variable
x = tf.placeholder(tf.float32, [None, 784])  # 2-D Tensor, where 784 = 28*28 array for the MNIST image data

# Initializing the 'Weights' and 'bias'
W = tf.Variable(tf.zeros([784, 10]))  # 784-D image vectors * W -> 10-dimensional vectors of evidence
b = tf.Variable(tf.zeros([10]))  # [10] -> so it can be added to the W*x output

# Defining the Softmax Regression model
# y = tf.nn.softmax(tf.matmul(x, W) + b)  <- Tutorial page applies tf.nn.softmax() but this brings down accuracy
y = tf.matmul(x, W) + b

# Initializing 'y_' as another placeholder for 'cross-entropy'
y_ = tf.placeholder(tf.float32, [None, 10])

# Implementiing 'cross-entropy' to measure inefficency of predictions
# From tutorial -> cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# From sample code:
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

print("Starting training stage...")

# Using Gradient Descent (with learning rate = 0.5) to minimize cross-entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initializing all the variable so far
init = tf.initialize_all_variables()

# Launching the completed model in a Session
sess = tf.Session()
sess.run(init)

print("Training the model 1000 times...")

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("Evaluating the model...")

# Evaluating whether the predicated label vs. the actual label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Casting booleans to floats and then taking the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Resulting accuracy of the model (should be ~92%)
print("Accuracy: " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

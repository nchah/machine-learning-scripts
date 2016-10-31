#!/usr/bin/env python

"""
Tutorial link: https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html

# Run after activating the tensorflow environment
$ python mnist_softmax_expert.py

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

# Possible to set additional configurations
# sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
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

# Resulting accuracy of the model
print("Accuracy: " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


print("Implementing Multilayer Convolutional Network to improve accuracy.")

# Implementing a Multilayer Convolutional Network

# Weight Initialization

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 1st Convolutional layer

# This layer will compute 32 features for each 5x5 patch
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape x to a 4d tensor. 2nd, 3rd dimensions = width and height, 4th = number of color channels
x_image = tf.reshape(x, [-1,28,28,1])

# convoluate x_image with the Weight tensor, add bias, apply ReLU function, and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolutional layer

# This layer has 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer

# Image has been reduced to 7x7 so now add 1024 neuron layer to process full image
# reshape tensor from the pooling layer into a batch of vectors, multiply by weight matrix, add bias, and apply ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout

# Apply dropout before the readout layer to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer

# The final layer for the softmax regression
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print("Training the model...")


# This step does 20,000 training iterations and may take up to 30 mins depending on local setup
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


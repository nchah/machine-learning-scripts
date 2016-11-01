#!/usr/bin/env python
"""
Source: https://www.oreilly.com/learning/hello-tensorflow
"""

import tensorflow as tf

# 1. TensorFlow Graph - first creating the computation graph

# Starting off with the default graph
graph = tf.get_default_graph()

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')

# 2. Sessions - computing in Sessions

# Initialize all variables and start the session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

summary_writer = tf.train.SummaryWriter('log_simple_graph/', sess.graph)


# View TensorBoard by running
# $ tensorboard --logdir=log_simple_graph
#
# and going to localhost:6006/#graphs in a web browser

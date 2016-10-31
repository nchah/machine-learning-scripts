#!/usr/bin/env python

import tensorflow as tf

# 1. TensorFlow Graph - first creating the computation graph

# Starting off with the default graph
graph = tf.get_default_graph()

# Currently there are no operations specified in the graph
graph.get_operations()

# This sets a Constant of 1.0 as a node/operation in the graph
input_value = tf.constant(1.0)

# Showing the protocol buffer for that first node
operations = graph.get_operations()
operations[0].node_def


# 2. Sessions - computing in Sessions

# Start the session and print out the input_value Constant
sess = tf.Session()
sess.run(input_value)


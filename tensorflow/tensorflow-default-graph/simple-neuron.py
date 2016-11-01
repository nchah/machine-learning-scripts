#!/usr/bin/env python
"""
Source: https://www.oreilly.com/learning/hello-tensorflow
"""

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

# Setting another variable
weight = tf.Variable(0.8)

# There were 4 operations added from that variable
for op in graph.get_operations():
    print(op.name)

# Following this simple calculation
output_value = weight * input_value


# 2. Sessions - computing in Sessions

# Initialize all variables and start the session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(output_value)


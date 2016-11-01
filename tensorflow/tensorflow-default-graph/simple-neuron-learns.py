#!/usr/bin/env python
"""
Source: https://www.oreilly.com/learning/hello-tensorflow
"""

import tensorflow as tf

# 1. TensorFlow Graph - first creating the computation graph

# Starting off with the default graph
graph = tf.get_default_graph()

# Setting the Ops
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')

# Assume that the correct value == 0.0
y_ = tf.constant(0.0)

# Loss is the "wrongness" of the model
# The model should lessen the square of the diff b/w current output and desired output.
loss = (y - y_)**2

# Adding an optimizer to add the "Learning" component
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)


# 2. Sessions - computing in Sessions

# Initialize all variables and start the session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

"""

grads_and_vars = optim.compute_gradients(loss)
sess.run(tf.initialize_all_variables())
sess.run(grads_and_vars[1][0])
# > 1.6 as intended

# Apply the gradient, finishing the backpropagation
sess.run(optim.apply_gradients(grads_and_vars))
sess.run(w)
"""

# Combining this into a single train_step
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for i in range(100):
    sess.run(train_step)

print(sess.run(y))
# > 0.004... bringing it closer to 0.0


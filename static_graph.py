""" Implementing the 2 layer network defined in basics.py
using tensorflow, for understanding the difference between
dynamic graph (pytorch) and static graph (tensorflow).
"""

import tensorflow as tf
import numpy as np

# ################################################################### #
# Implementing a 2 layer network with 1 hidden layer using tensorflow #
# ################################################################### #

N = 64          # batch size
D_in = 1000     # input dimension
H = 100         # hidden dimension
D_out = 10      # output dimension

# define the placeholders for the input and target data
# these will be filled when the graph is executed.
x = tf.placeholder(dtype=tf.float32, shape=(None, D_in), name="inputs")
y = tf.placeholder(dtype=tf.float32, shape=(None, D_out), name="targets")

# create the weights and initialize them with random data
# A tensorflow variable persists its value across executions of graph.
w1 = tf.Variable(tf.random_normal((D_in, H)), name="w1")
w2 = tf.Variable(tf.random_normal((H, D_out)), name="w2")

# forward pass
# this doesn't perform any numeric operations
# it merely sets up the computational graph that we will execute later.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# loss
loss = tf.reduce_sum((y_pred - y) ** 2.0)

# backward propagation
# compute gradients
w1_grad, w2_grad = tf.gradients(loss, [w1, w2])

learning_rate = 1e-6

# update the weights
# to update the weights we need to evaluate new_w1 and new_w2 when executing the graph.
# In tensorflow updating the weights is part of the computational graph.
# In pytorch this happens outside the computational graph.
new_w1 = w1.assign(w1 - (learning_rate * w1_grad))
new_w2 = w2.assign(w2 - (learning_rate * w2_grad))

print('-------------------------------------')
print("Training the network using tensorflow")
print('-------------------------------------')

with tf.Session() as sess:
    # initialize all the variables with their initializers.
    sess.run(tf.global_variables_initializer())

    # create the data
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    for epoch in range(500):
        # feed the x_value and y_value to x, y placeholders.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
        if epoch % 50 == 0:
            print(f"epoch : {epoch}, loss : {loss_value}")

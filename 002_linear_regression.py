import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 用 numpy 亂數產生 100 個點，並且
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.) 
# 等等 tensorflow 幫我們慢慢地找出 fitting 的權重值
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)


# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

loop_number=200
# Fit the line.
for step in range(loop_number+1):
    sess.run(train)

    if step % 10 == 0:
        #plt.gcf().clear()#clear figure
        plt.clf()
        print(step, sess.run(W), sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(y), label='Fitted line')
        plt.legend()
        if step == loop_number:
            plt.show()
        else:
            plt.show(False)
            plt.pause(0.1)#delay 0.11 second

       

# Learns best fit is W: [0.1], b: [0.3]
#use draw instead of show for noblock 

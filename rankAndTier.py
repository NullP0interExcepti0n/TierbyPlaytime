import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

playTime = np.transpose(data[0])
rank = np.transpose(data[1])

W = tf.Variable(tf.random_uniform([1], 0, 20000))
b = tf.Variable(tf.random_uniform([1], 1, 2000000))

X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000001)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(500):
		_, cost_val = sess.run([train_op, cost], feed_dict = {X: playTime, Y: rank})
		print(step, cost_val, sess.run(W), sess.run(b))

	print("\n=== Test ===")
	print("Play Time : 2100hrs, Rank :", sess.run(hypothesis, feed_dict={X: 2100}))
import tensorflow as tf

playTime = [2000, 3822, 3580, 2465, 1046]
rank = [200000-5000, 200000-720, 200000-1452, 2000000-480995, 2000000-124625]

W = tf.Variable(tf.random_uniform([1], 0.01, 1500))
b = tf.Variable(tf.random_uniform([1], 100, 10000))

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
	print("X:3000, Y:", sess.run(2000000 - hypothesis, feed_dict={X: 3000}))
	print("X:2800, Y:", sess.run(2000000 - hypothesis, feed_dict={X:2800}))
	print("X:2500, Y:", sess.run(2000000 - hypothesis, feed_dict={X:2500}))
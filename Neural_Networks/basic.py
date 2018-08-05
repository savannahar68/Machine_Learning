import tensorflow as tf

x = tf.constant(23)
y = tf.constant(35)

result  = tf.multiply(x, y) #for matrix multiplication use matmul

with tf.Session() as sess:
	print(sess.run(result))
'''
http://blog.csdn.net/weixin_35653315/article/details/74632110
tensorflow ExponentialMovingAverage
'''
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np

W_LEN = 2
w = tf.Variable(tf.zeros((W_LEN, 1)), name='weight')
b = tf.Variable(0.0, name='bias')
x = tf.placeholder(dtype=tf.float32, shape=[None, None])
Y = tf.placeholder(dtype=tf.float32, shape=[None, ])
y = tf.matmul(x, w) + b;

loss = tf.reduce_mean(tf.square(y - Y))
opt = tf.train.MomentumOptimizer(0.01, 0.99)
grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
update_op = opt.apply_gradients(grads_and_vars)
ema = tf.train.ExponentialMovingAverage(0.25)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([update_op]):
    train_op = tf.group(ema_op)
loss = control_flow_ops.with_dependencies([train_op], loss)

saver = tf.train.Saver()
save_dir = '/tmp/'
save_path = save_dir + 'lr_model'


def get_batches(num_batches, N=2, batch_size=1):
    w = np.arange(W_LEN);
    b = np.mean(w);
    x = np.random.randn(N, W_LEN)
    x = np.asarray(x, dtype=np.float32)
    y = np.dot(x, w.transpose()) + b
    batch_idx = 0;
    count = 0;
    while count < num_batches:
        count += 1
        batch_idx += 1;
        start = batch_idx * batch_size
        if start >= N:
            start = 0
        end = start + batch_size
        yield x[start: end], y[start: end]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for x_data, y_label in get_batches(num_batches=100):
        sess.run([loss], feed_dict={x: x_data, Y: y_label})
        step += 1
    saver.save(sess, save_path)

ckpt_reader = tf.train.NewCheckpointReader(save_path)
ckpt_vars = ckpt_reader.get_variable_to_shape_map()
print("variables in checkpoint:")
for name in ckpt_vars:
    var = ckpt_reader.get_tensor(name)
    print(  '\t', name, var)

print("==============================================")
print("variables restored not from ExponentialMovingAverage:")
with tf.Session() as sess:
    saver.restore(sess, save_path)
    for v in tf.trainable_variables():
        print(        '\t', v.name, v.eval())

print("==============================================")
print("variables restored from ExponentialMovingAverage:")
with tf.Session() as sess:
    variables_to_restore = ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)
    for v in tf.trainable_variables():
        print( '\t', v.name, v.eval())

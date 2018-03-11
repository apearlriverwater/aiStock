'''
    基于mnist的最佳实践进行修改  20180311
'''
import tensorflow as tf
import stock_inference
import os
import time
import stock_class  #提供stock训练需要的数据
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="stock_model/"
MODEL_NAME="stock_model"

'''
    stock_class模仿mnist最佳实践的mnist类进行设计，提供训练需要的数据
'''
def train():
    x = tf.placeholder(tf.float32, [None, stock_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, stock_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = stock_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('prediction'):
        # 绝对精度  完全相等的预测结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        #ones= tf.constant(1.0, shape=[g_max_stage])
        #loss=tf.reduce_sum(1-accuracy)

        reward_prediction = tf.argmax(y, 1)
        #reward_prediction = tf.reduce_mean(tf.cast(reward, tf.float32))

        # 误差在1个单位增益范围内的结果
        valid_prediction = tf.less_equal(
            abs(tf.argmax(y, 1) - tf.argmax(y_, 1)),stock_class.g_stage_rate)

        error=tf.reduce_mean(tf.cast(tf.abs( tf.argmax(y, 1)-tf.argmax(y_, 1)),tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
        tf.summary.scalar('valid_accuracy', valid_accuracy)
        tf.summary.scalar('error', error)

        # 误差在两个单位增益范围内的结果
        valid_prediction2 = tf.less_equal(
            abs(tf.argmax(y, 1) - tf.argmax(y_, 1)), 2 * stock_class.g_stage_rate)
        valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))
        tf.summary.scalar('valid_accuracy2', valid_accuracy2)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))

        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            stock_class.get_stock_count(), LEARNING_RATE_DECAY,
            staircase=True)

        #loss_mean = tf.reduce_mean(tf.cast(loss, tf.float32))
        tf.summary.scalar('loss', loss)


        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        merged = tf.summary.merge_all()


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for stock in stock_class.get_stock_list():
            train_writer = tf.summary.FileWriter(
                os.path.join(MODEL_SAVE_PATH, stock), sess.graph)
            stock_class.create_market_data(stock)
            next_stock=False

            i=0
            while not next_stock:
                xs, ys,next_stock = stock_class.next_batch()
                _, loss_value, step,summary = sess.run([train_op, loss, global_step,merged],
                                               feed_dict={x: xs, y_: ys})
                i+=1
                if i%50==0:
                    train_writer.add_summary(summary, i)

            if next_stock:
                print(" [%s]loss on training [%s]  is [%g]." %
                      (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                       stock, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()



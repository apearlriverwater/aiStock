'''
    基于mnist的最佳实践进行修改  20180311
'''
import tensorflow as tf

import os
import time
import stock_class  #提供stock训练需要的数据
import stock_inference
import stock_eval

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="/zyj/model/"
MODEL_NAME="stock_model"

'''
2018-03-13 执行完毕后，第一个股票的模型参数会被自动删除，原因不明,原因是保留
    几个最新的模型文件导致的结果。
    似乎是执行滑动平滑操作后tf自动删除旧数据。
    考虑输出两种模型： 基于特定标的的模型
                    基于组合的模型，需分别验证其有效性    
    stock_class模仿mnist最佳实践的mnist类进行设计，提供训练需要的数据
    
'''
def train():
    x = tf.placeholder(tf.float32, [None, stock_class.g_input_columns], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, stock_class.g_max_stage], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = stock_inference.inference(x, regularizer,stock_class.g_input_columns,stock_class.g_max_stage)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('prediction'):
        # 绝对精度  完全相等的预测结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        #reward_prediction = tf.argmax(y, 1)
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
            abs(tf.argmax(y, 1) - tf.argmax(y_, 1)), 1)

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
            stock_class.g_look_back_weeks, LEARNING_RATE_DECAY,
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
        last_acc,last_acc1=-10,-10
        train_count=0
        no_change_count=0
        for stock in stock_class.get_stock_list():
            train_count += 1
            train_writer = tf.summary.FileWriter(
                os.path.join(MODEL_SAVE_PATH,stock_class.STOCK_BLOCK+'.BLK/'+ stock), sess.graph)
            stock_class.create_market_data(stock)

            if len(stock_class.train_x) == 0:
                continue

            next_stock=False

            i=0
            while not next_stock:
                xs, ys,next_stock = stock_class.next_batch()
                _, loss_value, step,summary = sess.run([train_op, loss, global_step,merged],
                                               feed_dict={x: xs, y_: ys})
                i+=1
                if i%50==0:
                    #train_writer.add_summary(summary, i)
                    pass

            if next_stock:
                print("[%s]loss on training [%s]  is [%g]." %
                      (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                       stock, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                           stock_class.STOCK_BLOCK + '.BLK/'+stock_class.STOCK_BLOCK),
                           global_step=global_step)

                if train_count%2==0:
                    acc,acc1=stock_eval.evaluate(False)
                    if abs(last_acc-acc)==0 and abs(last_acc1-acc1)==0:
                        no_change_count+=1
                        if no_change_count>5:
                        #模型参数已确定，没有进一步训练的价值
                            print('[%s]模型精度已连续5个标的未变化，stop model training!!!'%(
                                time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
                            break
                    else:
                        no_change_count=0
                        last_acc = acc
                        last_acc1 = acc1

                #检查是否已训练到位  精度不变时继续训练没有意义

def main(argv=None):
    train()

if __name__ == '__main__':
    print("=====[%s] start training" %
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    tf.app.run()
    print("=====[%s] stop training" %
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))


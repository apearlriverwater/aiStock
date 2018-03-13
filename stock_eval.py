import time
import tensorflow as tf
import stock_inference
import stock_train
import stock_class

def evaluate(for_ever=True):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, stock_class.g_input_columns], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, stock_class.g_max_stage], name='y-input')


        y = stock_inference.inference(x, None,stock_class.g_input_columns,stock_class.g_max_stage)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction1 =tf.less_equal(tf.abs(tf.argmax(y, 1)-tf.argmax(y_, 1)),1)
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(stock_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)


        with tf.Session() as sess:
            last_ok=-10
            while True:
                total_ok=0
                total_ok1 = 0
                total=0

                for stock in stock_class.get_stock_list():
                    stock_ok = 0
                    stock_ok1=0
                    stock_total = 0

                    #print(" [%s] [%s]loss on backtest." %
                    #      (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),stock))
                    ckpt = tf.train.get_checkpoint_state(stock_train.MODEL_SAVE_PATH)
                    #ckpt = tf.train.get_checkpoint_state(os.path.join(
                    #    stock_train.MODEL_SAVE_PATH, stock_class.STOCK_BLOCK + '.BLK'))
                    if ckpt and ckpt.model_checkpoint_path:
                        try:
                            saver.restore(sess, ckpt.model_checkpoint_path)
                            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                            stock_class.create_market_data(stock,stock_class.g_train_stopDT,
                                            stock_class.g_backtest_stopDT,stock_class.g_look_back_weeks)
                            next_stock = False

                            while not next_stock:
                                xs, ys, next_stock = stock_class.next_batch()
                                validate_feed = {x: xs, y_: ys}

                                accuracy_score,accuracy_score1 = sess.run([accuracy,accuracy1], feed_dict=validate_feed)
                                #print("accuracy = %g,accuracy1 = %g"
                                #      % (accuracy_score, accuracy_score1))

                                total+=1
                                stock_total+=1

                                if accuracy_score>0.5:
                                    total_ok+=1
                                    stock_ok+=1

                                if accuracy_score1>0.5:
                                    stock_ok1+=1
                                    total_ok1 += 1

                                if next_stock and stock_total>0:
                                    #print("     [%s]accuracy = %.2f%%,accuracy1 = %.2f%%"
                                    #      % (stock,stock_ok*100/stock_total,stock_ok1*100/stock_total))
                                    pass

                    #time.sleep(EVAL_INTERVAL_SECS)
                        except:
                            pass

                if total>0:
                    print("[%s]BLOCK [%s] step=[%s]  total accuracy = %.2f%%,accuracy1 = %.2f%%"
                           % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                              stock_class.STOCK_BLOCK,global_step, total_ok * 100 / total, total_ok1 * 100 / total))

                #定期自动评估模型的精度，如果精度没有变化，终止评估
                if abs(last_ok-total_ok)<=1 and not for_ever:
                    print('stop backtest!')
                    break
                else:
                    last_ok = total_ok
                    time.sleep(60*3)

            return total_ok , total_ok1


if __name__ == '__main__':
    evaluate()
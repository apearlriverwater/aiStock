# coding=utf-8

import tensorflow as tf
import datetime
import time
import gmTools_v2 as gmTools
import nn_model as nn


#沪深300（SHSE.000300）、SHSE.000947 	内地银行  SHSE.000951 	300银行
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000016'

MAX_HOLDING=5
MAX_STOCKS=20
learning_rate = 0.001
dropout = 0.9
# 策略参数
max_holding_days = 15
trade_minutes = 240

    
g_train_startDT=datetime.datetime.strptime('2016-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')  # oldest start 2015-01-01
g_train_stopDT=datetime.datetime.strptime('2018-03-01 09:00:00', '%Y-%m-%d %H:%M:%S')
g_backtest_stopDT=datetime.datetime.strptime('2018-04-01 09:00:00', '%Y-%m-%d %H:%M:%S')


def train_model(week=30,look_back_weeks=100,debug=False):
    g_week_in_trade_day = int(trade_minutes / week)
    g_look_back_weeks = max(10, g_week_in_trade_day * 2) * 10  # 回溯分析的周期数

    g_log_dir = '/01deep-ml/logs/w{0}hold{1}days'.format(week, max_holding_days)
    # 定义网络参数  setup_tensor(sess, STOCK_BLOCK, g_week, 0)
    with tf.name_scope("model_vars"):
        model_code = tf.Variable(gmTools.STOCK_BLOCK, 
                dtype=tf.string, trainable=False, name="model_code")
        
        model_week = tf.Variable(week, dtype=tf.int32, 
                trainable=False, name="model_week")
        
        model_last_utc = tf.Variable(0, dtype=tf.int64,
                trainable=False, name="model_last_utc")
        
        model_next_train_utc = tf.Variable(0, dtype=tf.int64,
                trainable=False, name="model_next_train_utc")

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
                gmTools.g_input_columns], name='x_input')
        y = tf.placeholder(tf.float32, 
                [None, gmTools.g_max_stage], name='y_input')

    nn_hidden_nodes = 800
    hidden = nn.nn_layer(x,gmTools.g_input_columns,
                nn_hidden_nodes, 'layer1')

    with tf.name_scope('g_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        # tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden, keep_prob)

    y1 = nn.nn_layer(dropped, nn_hidden_nodes,
                     gmTools.g_max_stage, 'layer2', act=tf.identity)

    """ 
    这里使用tf.nn.softmax_cross_entropy_with_logits()
    对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy。 
    计算平均损失，并使用tf.summary.saclar进行统计汇总。 
    """
    with tf.name_scope('prediction'):
        # 绝对精度  完全相等的预测结果
        correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        # ones= tf.constant(1.0, shape=[g_max_stage])
        # loss=tf.reduce_sum(1-accuracy)

        reward_prediction = tf.argmax(y1, 1)

        # 误差在1个单位增益范围内的结果
        valid_prediction = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 1)
        error = tf.reduce_mean(tf.cast(tf.abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
        tf.summary.scalar('valid_accuracy', valid_accuracy)
        tf.summary.scalar('error', error)

        # 误差在两个单位增益范围内的结果
        valid_prediction2 = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 2)
        valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))
        tf.summary.scalar('valid_accuracy2', valid_accuracy2)

        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
        # diff = tf.nn.softmax_cross_entropy_with_logits(logits=prediction_mean, labels=true_mean)
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    """ 
    使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray， 
    再使用tf.summary.scalar对accuracy进行统计汇总。 
    train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)
    """
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(nn.learning_rate).minimize(cross_entropy)

    merged = tf.summary.merge_all()
    # 自动生成工程需要的文件目录
    # train_writer = tf.summary.FileWriter(g_log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(g_log_dir + '/test')
    test_figure = tf.summary.FileWriter(g_log_dir + '/fig')

    securities=gmTools.get_block_stock_list(STOCK_BLOCK)[:MAX_STOCKS]

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=len(securities)+2)
    sess.run(tf.assign(model_week, week))

    ii=0
    total_count=0
    last_acc1=-10
    last_acc=-10
    no_change_count=0

    for stock in securities:
        model_path = g_log_dir + '/model/'+STOCK_BLOCK+'/'
        model_file = tf.train.latest_checkpoint(model_path)

        if model_file:
            try:
                saver.restore(sess, model_file)
                week, code, last_utc = sess.run([model_week, model_code, model_last_utc])
                # code string ,返回是bytes类型，需要转换
                print("restore from model code=%s,week=%d,last_utc %d" % (
                    code, week, last_utc))
            except:
                pass

        ii += 1
        print("\n[%s]start training model [%s] %.2f%%"%(stock,
                time.strftime('%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())),
                ii*100/len(securities)))

        g_market_train_data=gmTools.create_market_data(stock=stock,
                           start_DateTime=g_train_startDT,
                           stop_DateTime=g_train_stopDT ,
                           week=week, look_back_weeks=look_back_weeks)

        g_max_step = len(g_market_train_data)
        print('log dir:%s , total items :%d' % (g_log_dir, g_max_step))

        if g_max_step<=g_week_in_trade_day:
            continue

        train_count=int(g_max_step)

        g_current_train_stop = 0
        i=g_current_train_stop

        print("training %s" % (g_market_train_data.iloc[0,0].strftime('%Y-%m-%d %H:%M:%S')))

        train_writer = tf.summary.FileWriter(model_path+'/'+stock, sess.graph)

        for i in range( look_back_weeks,train_count - 1):
            xs,ys = gmTools.feed_dict(True)
            feed_dict_data={x:xs,y:ys,keep_prob:dropout}
            #
            summary,_,reward= sess.run(
                [merged,train_step,reward_prediction],
                feed_dict=feed_dict_data)

            if debug and total_count% 20==0:
                train_writer.add_summary(summary, i)

            total_count += 1

        #评估训练的效能  同一模型不能打开两次
        '''
        if ii/len(securities)>0.3:
            acc, acc1,_ = backtest_model()
            if abs(last_acc - acc) == 0 and abs(last_acc1 - acc1) == 0:
                no_change_count += 1
                if no_change_count > 5:
                    # 模型参数已确定，没有进一步训练的价值
                    print('[%s]模型精度已连续5个标的未变化，stop model training!!!' % (
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                    break
            else:
                no_change_count = 0
                if acc>0:
                    last_acc = acc
                    last_acc1 = acc1
        '''

        g_market_train_data = 0


        sess.run(tf.assign(model_code,stock))
        saver.save(sess, model_path + stock+'/model.ckpt')  # save模型  for stock

        sess.run(tf.assign(model_code,STOCK_BLOCK))
        saver.save(sess, model_path +STOCK_BLOCK+ '.BLOCK/model.ckpt')  # save模型  for all

    sess.close()
    print('total train %d steps'%(total_count))




'''
    基于特定股票的回测模型恢复
'''
def restore_stock_model(stock,
    sess,saver,model_week, model_code):

    paths=g_log_dir + '/model/'+STOCK_BLOCK+'/'
    model_paths =[paths+stock,paths+STOCK_BLOCK+'.BLOCK']
    ret=False
    for model_path in model_paths:
        model = tf.train.latest_checkpoint(model_path)

        if model:
            try:
                saver.restore(sess, model)
                week, code = sess.run([model_week, model_code])
                # code string ,返回是bytes类型，需要转换
                print("restore from model code=%s,week=%d" % (
                    code, week))
                ret=True
                break
            except:
                #print("restore from [%s] error"%(model_path))
                pass

    return  ret



if __name__ == '__main__':
    #训练模型
    train_model()

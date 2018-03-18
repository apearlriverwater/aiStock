# coding=utf-8

import tensorflow as tf
import nn_model as nn
import gmTools_v2 as gmTools
import train_stock
import time
import datetime

MAX_STOCKS=20
MAX_HOLDING=5


#沪深300（SHSE.000300）、SHSE.000947 	内地银行  SHSE.000951 	300银行
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000016'
'''
    由于掘金返回批量股票K线数据很慢，采用集中获取买卖点信息后统一处理，
    自行计算买卖数据，自行统计盈利情况
'''
def backtest_model(securities,week,look_back_weeks,
                    startDT ,stopDT,hold_days):

    global valid_accuracy,accuracy,valid_accuracy2,\
        reward_prediction,x,y,keep_prob

    print("\nstart backtest_model [%s]" % (
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    # 策略参数
    trade_minutes = 240
    week_in_trade_day = int(trade_minutes / week)

    g_log_dir = '/01deep-ml/logs/w{0}hold{1}days'.format(week, hold_days)

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
    hidden = nn.nn_layer(x, gmTools.g_input_columns,
                         nn_hidden_nodes, 'layer1')

    with tf.name_scope('g_dropout'):
        keep_prob = tf.placeholder(tf.float32)
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

        reward_prediction = tf.argmax(y1, 1)

        # 误差在1个单位增益范围内的结果
        valid_prediction = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 1)
        valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))

        valid_prediction2 = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 2)
        valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # restore model parameters

    acc_list=[]
    acc, acc1, total=0,0,0
    ii = 0
    #bacttest_start_date = gmTools.get_backtest_start_date(g_train_stopDT,
    #                    int(1.5 * g_look_back_weeks / g_week_in_trade_day))
    bacttest_start_date=gmTools.get_backtest_start_date(startDT,
                    int(1.5*look_back_weeks/week_in_trade_day))

    for stock in securities:
        #try:
            print("\n[%s]start backtesting [%s] %.2f%%" % (stock,
                   time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   ii * 100 / len(securities)))
            ii += 1
            acc,acc1,acc2,total=backtest_stock(stock=stock, sess=sess,saver=saver,
                           week=week,
                           look_back_weeks=look_back_weeks,
                           hold_days=hold_days,
                           read_start_DT=bacttest_start_date,
                           startDT =startDT,
                           stopDT=stopDT)

            acc_list.append([stock,acc,acc1,acc2,total])
        #except:
        #    print('error in backtest_model, doing [%s]'%(stock))

    acc_total=0
    acc1_total=0
    acc2_total = 0
    totals=0
    for (stock,acc, acc1 ,acc2, total) in acc_list:
        if total>0:
            print("[%s] acc=%.2f%%,acc1=%.2f%%,acc2=%.2f%%" % (
                stock, acc * 100 / total, acc1 * 100 / total, acc2 * 100 / total))

        acc_total+=acc
        acc1_total += acc1
        acc2_total += acc2
        totals+=total

    if totals>0:
        print("total acc=%.2f%%,acc1=%.2f%%,acc2=%.2f%%" % (
                acc_total * 100 / totals, acc1_total * 100 / totals, acc2_total * 100 / totals))

    sess.close()

    return acc,acc1,acc2,total

'''
    基于特定股票的回测
'''
def backtest_stock(stock,sess,saver, week,
       look_back_weeks,hold_days,
        read_start_DT,startDT ,stopDT):

    acc = 0
    acc1 = 0
    acc2 = 0
    total = 0
    log_dir = '/01deep-ml/logs/w{0}hold{1}days/model/{2}/'.format(
        week, hold_days,STOCK_BLOCK)

    if restore_stock_model(stock,sess,saver,log_dir)==False:
        return acc, acc1,acc2, total

    g_market_train_data=gmTools.create_market_data(stock=stock,
                   start_DateTime=read_start_DT ,
                   stop_DateTime=stopDT ,
                   week=week, look_back_weeks=look_back_weeks)

    g_max_step = len(g_market_train_data)
    if g_max_step < look_back_weeks:
        g_market_train_data=0
        return acc, acc1, acc2, total


    train_count =g_max_step  # int(g_max_step / look_back_weeks)

    g_current_train_stop = 0
    i = g_current_train_stop
    try:
        print("[%s]backtesting %s" % (
            g_market_train_data.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S'),stock))
    except:
        print('[%s]error backtesting %s ' % (
            g_market_train_data.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S'),stock))
        pass

    last_reward = [5, 5]
    buy_on = False  # 连续出现的buy信号不处理
    buy_point = 0
    code_4_buy = []
    code_4_sell = []
    # detect buy-sell signal
    # 最新的hold_weeks个周期由于没有数据，回测时无法使用但可作预测买卖点

    buy_date=0
    for i in range(look_back_weeks, train_count - 1):
        #跳过回测范围外的数据
        if g_market_train_data.iloc[i, 0]<startDT:
            gmTools.feed_dict(True)
            continue

        xs, ys = gmTools.feed_dict(True)
        feed_dict_data = {x: xs, y: ys, keep_prob: 1}
        reward,valid_acc,valid_acc1,valid_acc2 = sess.run(
            [reward_prediction,accuracy,valid_accuracy,valid_accuracy2],
            feed_dict=feed_dict_data)

        if valid_acc>0.5:
            acc+=1

        if valid_acc1>0.5:
            acc1+=1

        if valid_acc2>0.5:
            acc2+=1

        total+=1

        #买入当日不再判断买卖点
        if buy_on and g_market_train_data.iloc[i, 0].date()==buy_date:
            continue

        # todo 简单判断未来走势未必合理，是否考虑看均线趋势？
        index = 0
        last_reward = last_reward + reward.tolist()

        if True:
            #todo  待建立买点判断模型  用当前的实际收益情况判断是否适合买入
            # 利用close真值判断i
            if not buy_on and last_reward[index - 1] >= gmTools.BUY_GATE \
               and last_reward[index - 2] >= gmTools.BUY_GATE:
                #利用当前的价格走势确定是否为有效买点
                #todo maup,rsi,macd??  use tf model to make decision
                kdata=g_market_train_data.iloc[ i- look_back_weeks:i, :]

                if gmTools.can_buy(kdata):
                    buy_on = True
                    buy_point = i
                    buy_date=g_market_train_data.iloc[buy_point, 0].date()
                    code_4_buy.append(
                          {'code': stock,
                           'time': g_market_train_data.iloc[buy_point, 0],
                           'price': g_market_train_data.iloc[ i, 1],
                           'reward': last_reward[index - 1]})

            if buy_on :
                #todo  待建立卖点判断模型
                kdata = g_market_train_data.iloc[i - look_back_weeks:i, :]

                if gmTools.can_sell(kdata) \
                  or i  - buy_point > hold_days*week_in_trade_day:  # arrive max holding weeks

                    buy_on = False
                    code_4_sell.append(
                        {'code': stock,
                        'time': g_market_train_data.iloc[i, 0],
                        'price': g_market_train_data.iloc[ i, 1],
                        'reward': last_reward[index - 1]})

        last_reward = last_reward[-2:]



    #if total>0:
    #    print("[%s] acc=%.2f%%,acc1=%.2f%%"%(stock,acc*100/total,acc1*100/total))

    g_market_train_data = 0
    g_normalized_data = 0
    g_max_step = 0
    train_x = 0
    train_y = 0

    # 对买卖列表按时间顺序进行排序，按时间段进行买卖点分析
    code_4_buy.sort(key=lambda i: (i['time'], i['reward']))
    code_4_sell.sort(key=lambda i: (i['time'], i['reward']))

    # process BS point
    buy_index = 0
    sell_index = 0
    holding = []
    holdings = []
    amount = 1e6
    is_sell = True

    bs_index_changed = False

    # process buy sell point
    while len(code_4_sell[sell_index:]) > 0 or len(code_4_buy[buy_index:]) > 0:
        bs_time = datetime.datetime.fromtimestamp(int(time.time()))
        # from the min time on
        try:
            if buy_index < len(code_4_buy):
                bs_time = code_4_buy[buy_index]['time']
                is_sell = False

            if sell_index < len(code_4_sell):
                bs_time = min(code_4_sell[sell_index]['time'], bs_time)
                is_sell = True

            buy_list = []
            for item in code_4_buy[buy_index:]:
                if item['time'] == bs_time:
                    buy_list.append(item)
                    buy_index += 1
                    bs_index_changed = True
                else:
                    break

            if len(buy_list) > 0 and len(holding) < MAX_HOLDING:
                buy_list.sort(key=lambda i: i['reward'], reverse=True)
                buy_list = buy_list[:MAX_HOLDING]

                for item in buy_list:
                    if len(holding) >= MAX_HOLDING:
                        break
                    elif not item['code'] in holdings:  # buy only once
                        money = amount / (MAX_HOLDING - len(holding)) / 1.1
                        vol = int(money / (item['price'] * (1 + gmTools.BUY_FEE) * 100)) * 100
                        if vol > 0:
                            tmp = vol * item['price'] * (1 + gmTools.BUY_FEE)
                            amount -= tmp
                            holdings.append(item['code'])

                            start_datetime = bs_time \
                                +datetime.timedelta(days=- train_stock.max_holding_days*2)

                            must_sell_datetime = bs_time \
                                +datetime.timedelta(days=train_stock.max_holding_days * 4)

                            # TODO ADD K DATA TO HONGDING AND DETECT LOSS
                            k_data = gmTools.read_kline(item['code'], int(week * 60),
                                    start_datetime, must_sell_datetime)  # k line 数据

                            cols = ['eob', 'close','volume','amount']
                            k_data = k_data[cols]

                            holding.append({'code': item['code'], 'price': item['price'],
                                            'vol': vol, 'time': bs_time, 'kdata': k_data})
                            print("[%s] buy %s vol=%d,price=%.2f,amt=%.2f,nav=%.2f" %
                                  (bs_time,item['code'], vol, item['price'], tmp, amount))

            if len(holding) > 0:
                sell_list = []
                for item in code_4_sell[sell_index:]:
                    if item['time'] == bs_time and item['code'] in holdings:
                        i = holdings.index(item['code'])

                        vol = holding[i]['vol']
                        tmp = vol * item['price'] * (1 - gmTools.SELL_FEE)
                        amount += tmp
                        # todo save bs point in the graph
                        gmTools.draw_bs_on_kline(holding[i]['code'],
                                holding[i]['kdata'], holding[i]['time'],
                                bs_time,week,log_dir)

                        print("[%s] sell %s vol=%d,bs price=[%.2f--%.2f],amt=%.2f,reward=%d,nav=%.2f" %
                              (bs_time,item['code'], vol, holding[i]['price'], item['price'], tmp,
                               int(item['price'] * 100 / holding[i]['price'] - 100), amount))

                        holding.pop(i)
                        holdings.pop(i)

                        sell_index += 1
                        bs_index_changed = True
                    else:
                        continue

            if bs_index_changed == False:
                if is_sell:
                    sell_index += 1
                else:
                    buy_index += 1
            else:
                bs_index_changed = False

        except:
            print('error backtest_model [%s]'%(stock))
            pass

    #todo 增加最后持仓的卖出处理
    if len(holding) > 0:
        i=0
        # todo save bs point in the graph
        gmTools.draw_bs_on_kline(holding[i]['code'],
                         holding[i]['kdata'], holding[i]['time'],
                         holding[i]['time'],week,log_dir)

        '''
        bs_time = holding[i]['time']
        vol = holding[i]['vol']
        tmp = vol * item['price'] * (1 - SELL_FEE)
        amount += tmp
        print("[%s] sell %s vol=%d,bs price=[%.2f--%.2f],amt=%.2f,reward=%d,nav=%.2f" %
              (bs_time, item['code'], vol, holding[i]['price'], item['price'], tmp,
               int(item['price'] * 100 / holding[i]['price'] - 100), amount))
        '''
    #sess.run(tf.assign(model_code,stock))
    #model_path = g_log_dir + '/model/' + stock
    #saver.save(sess, model_path + "/model.ckpt")  # save模型
    #sess.close()

    #print("\nstop backtest_model [%s]" % (
    #    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    return acc, acc1,acc2, total
'''
    基于特定股票的回测模型恢复
'''
def restore_stock_model(stock,sess,saver,paths):
    #paths=g_log_dir + '/model/'+STOCK_BLOCK+'/'
    model_paths =[paths+STOCK_BLOCK+'.BLOCK/',paths+stock+'/']
    ret=False
    for model_path in model_paths:
        model = tf.train.latest_checkpoint(model_path)
        if model:
            try:
                saver.restore(sess, model)
                ret=True
                break
            except:
                #print("restore from [%s] error"%(model_path))
                pass

    return  ret

if __name__ == '__main__':

    securities = gmTools.get_block_stock_list(gmTools.STOCK_BLOCK)[:MAX_STOCKS]

    # 策略参数
    week=60
    trade_minutes = 240
    week_in_trade_day = int(trade_minutes / week)
    look_back_weeks = max(10, week_in_trade_day * 2) * 10  # 回溯分析的周期数
    hold_days=15

    #训练模型
    train_startDT = datetime.datetime.strptime('2016-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')  # oldest start 2015-01-01
    train_stopDT = datetime.datetime.strptime('2018-03-01 09:00:00', '%Y-%m-%d %H:%M:%S')
    backtest_stopDT = datetime.datetime.strptime('2018-04-01 09:00:00', '%Y-%m-%d %H:%M:%S')
    backtest_model(securities,week,look_back_weeks,
                   train_stopDT,backtest_stopDT,hold_days)

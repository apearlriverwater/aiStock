# coding=utf-8

import tensorflow as tf
import numpy  as np
import gmTools_v2 as gmTools
import time
import  matplotlib.pyplot as plt
import pandas as pd
import datetime
import struct

""" 
2018-03-11:
    1)封装stock数据有关的接口，为深度学习提供预测、评估与实时预测所需数据
 
"""

#沪深300（SZSE.399300）、
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000016'

MAX_HOLDING=5
BUY_GATE=7
SELL_GATE=3
BUY_FEE=1E-4
SELL_FEE=1E-4
DAY_SECONDS=24*60*60
MAX_STOCKS=15

g_train_startDT=datetime.datetime.strptime('2015-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')  # oldest start 2015-01-01
g_train_stopDT=datetime.datetime.strptime('2018-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')
g_backtest_stopDT=datetime.datetime.strptime('2019-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')

g_max_step = 20000

#策略参数
g_week=30  #freqency
g_max_holding_days=15

g_input_columns=6
g_trade_minutes=240
g_week_in_trade_day=int(g_trade_minutes/g_week)
g_look_back_weeks=max(10,g_week_in_trade_day*2)  #回溯分析的周期数
g_max_holding_weeks=g_week_in_trade_day*g_max_holding_days  #用于生成持仓周期内的收益等级


g_max_stage=11  #持仓周期内收益等级
g_stage_rate=2  if g_week>30 else 1#持仓周期内收益等级差

g_log_dir = '/01deep-ml/logs/w{0}hold{1}days'.format(g_week,g_max_holding_days)

train_x, train_y = [], []
g_current_train_stop=0          #当前测试数据结束位置

g_test_stop=0                   #当前实时数据结束位置
g_stock_current_price_list=0

#标签数据生成，自动转成行向量 0-10共11级，级差1%
def make_stage(x):
    x=int(100*x/g_stage_rate)
    if abs(x)<5:
        x = x+5
    elif x>4:
        x = 10
    else:
        x = 0

    tmp = np.zeros(g_max_stage, dtype=np.int)
    tmp[x]=1
    return tmp

# 获取测试集  步进为time_step的测试数据块,
# 与训练数据格式不一致、处理方式也不一致，预测周期越长越不准确
# 数据块不完整时必须用0补充完整
def get_test_data(data, normalized_data,
    look_back_weeks=g_look_back_weeks):
    train_x, train_y = [], []
    start=look_back_weeks
    #for i in range(look_back_weeks, len(data)):
    for i in range(look_back_weeks,int(len(data))):
        x = normalized_data.iloc[start- look_back_weeks:start, :]
        y = data.iloc[start-look_back_weeks:start, -1]
        start+=1 #look_back_weeks

        train_x.append(x.values.tolist())

        #test_y.extend(y.values.tolist())
        train_y.append(y.values.tolist())

    return train_x, train_y

def create_market_data(stock,
        start_DateTime=g_train_startDT,
        stop_DateTime=g_train_stopDT,week=g_week,
        look_back_weeks=g_look_back_weeks):

    global  g_market_train_data,g_input_columns,\
        g_normalized_data,g_max_step,\
        train_x,train_y,g_current_train_stop

    g_current_train_stop = 0
    g_market_train_data=gmTools.read_kline(stock,int(week*60),
            start_DateTime,stop_DateTime,50000)    #训练数据
    if len(g_market_train_data)==0:
        return
    #预测look_back_weeks周期后的收益
    g_market_train_data['label']=g_market_train_data['close'].pct_change(look_back_weeks)
    g_market_train_data['label']=g_market_train_data['label'].shift(-look_back_weeks)
    #将数据总项数整理成g_max_holding_weeks的整数倍
    #tmp = len(g_market_train_data)%g_max_holding_weeks+g_max_holding_weeks
    #g_market_train_data =g_market_train_data[tmp:]
    g_market_train_data['label'] =g_market_train_data['label'].fillna(0)
    g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    data_tmp = g_market_train_data.iloc[:, 1:-1]
    #todo  加入其他的技术分析指标

    # 数据归一化处理

    mean = np.mean(data_tmp, axis=0)
    std = np.std(data_tmp, axis=0)
    g_normalized_data = (data_tmp - mean) / std  # 标准化

    g_input_columns=len(data_tmp.columns)

    cols=['eob', 'close','label']
    g_market_train_data = g_market_train_data[cols]
    g_max_step = len(g_market_train_data)

    #数据规整为look_back_weeks的整数倍
    remainder=len(g_market_train_data)%look_back_weeks
    g_market_train_data=g_market_train_data[remainder:]
    g_normalized_data = g_normalized_data[remainder:]

    train_x,train_y=get_test_data(g_market_train_data,
            g_normalized_data,look_back_weeks)


def create_market_last_n_data(stocks, count, stop_DateTime,
                       week=g_week, look_back_weeks=g_look_back_weeks):
    global g_stock_current_price_list, g_input_columns, \
        g_normalized_data, g_max_step, train_x, train_y

    market_train_data = gmTools.read_last_n_kline(stocks, int(week * 60),
        count, stop_DateTime)  # 训练数据

    g_max_step = len(market_train_data)

    if g_max_step == 0:
        return
    g_stock_current_price_list=[]
    train_x = []
    train_y = []

    #以排序后的股票代码为序保存g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    for kline in market_train_data:
        stock,kdata=kline['code'],kline['kdata']

        data_tmp = kdata.iloc[:, 1:]
        # todo  加入其他的技术分析指标

        # 数据归一化处理
        mean = np.mean(data_tmp, axis=0)
        std = np.std(data_tmp, axis=0)
        g_normalized_data = (data_tmp - mean) / std  # 标准化

        g_input_columns = len(data_tmp.columns)

        cols = ['eob', 'close']
        g_stock_current_price_list.append({'code':stock,'time_close':kdata[cols][-1:].values.tolist()})

        y=int(kdata['close'][len(kdata)-1]*100/kdata['close'][0]-100)
        train_x.append(g_normalized_data.values.tolist())
        #shape(?,g_max_tage)
        train_y.append([make_stage(y)])


def next_batch():
    global  g_current_train_stop

    xs = train_x[g_current_train_stop]
    ys = train_y[g_current_train_stop]

    g_current_train_stop+=1

    if g_current_train_stop>=len(train_x):
        next_stock=True
    else:
        next_stock=False

    return  xs, ys, next_stock

'''
    在价格走势图显示买卖点信息
'''
def draw_bs_on_kline(stock,kdata,buy_utc,sell_utc,week=g_week):
    # 以折线图表示结果 figsize=(20, 15)
    try:
        plt.figure()
        data=kdata['close'].values.tolist()

        plot = plt.plot(list(range(len(kdata))),
                    data, color='b', label='close')
        utclist=kdata['eob'].tolist()

        buy_time=buy_utc.strftime('%Y-%m-%d %H:%M:%S')
        sell_time=sell_utc.strftime('%Y-%m-%d %H:%M:%S')

        title = ' {2} week={3} \n [{0}--{1}]'.format(buy_time, sell_time, stock,week)

        plt.title(title)

        x=utclist.index(buy_utc)
        y=data[x]

        plt.annotate('buy', xy=(x, y),
                     xytext=(x * 1.1, y),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     )
        x = utclist.index(sell_utc)
        y = data[x]

        plt.annotate('sell', xy=(x, y),
                     xytext=(x * 0.9, y),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     )
    except:
        pass

    buy_time = buy_utc.strftime('%Y-%m-%d %H-%M-%S')
    sell_time = sell_utc.strftime('%Y-%m-%d %H-%M-%S')

    file = '{3}/{0}--{1}--{2}.png'.format(stock, buy_time, sell_time,g_log_dir + '/fig' )
    plt.savefig(file)
    plt.close()

def get_stock_count():
    return len(g_test_securities)

def get_stock_list():
    return g_test_securities

g_test_securities=gmTools.get_index_stock(STOCK_BLOCK)[:MAX_STOCKS]

if __name__ == '__main__':
    #训练模型
    #train_model()
    pass
'''
else:
    #回测期间不能反复初始化模型，否则会导致运行越来越慢
    model_path = g_log_dir

    sess = tf.InteractiveSession()
    setup_tensor(sess, '', g_week, 0)
    saver = tf.train.Saver()

    model_file = tf.train.latest_checkpoint(model_path)

    if model_file:
        try:
            saver.restore(sess, model_file)
            week, code = sess.run([model_week, model_code])
            # code string ,返回是bytes类型，需要转换
            print("restore from model code=%s,week=%d" % (
                code, week))
        except:
            print("restore from model error" )
            pass
'''
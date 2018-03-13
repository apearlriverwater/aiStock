# coding=utf-8

#import tensorflow as tf
import numpy  as np
import gmTools_v2 as gmTools
#import time
import  matplotlib.pyplot as plt
import pandas as pd
import datetime
#import struct
import talib

""" 
2018-03-13:
    加入技术指标无助于提高预测的准确性。
2018-03-11:
    1)封装stock数据有关的接口，为深度学习提供预测、评估与实时预测所需数据
 
"""

#沪深300（SHSE.000300）、
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000016'
g_input_columns=6

MAX_HOLDING=5
BUY_GATE=7
SELL_GATE=3
BUY_FEE=1E-4
SELL_FEE=1E-4
DAY_SECONDS=24*60*60
MAX_STOCKS=500
INDEX_LIST=['SHSE.000001','SHSE.000016','SHSE.000300','SZSE.399005','SZSE.399006','SZSE.399001']
g_train_startDT=datetime.datetime.strptime('2014-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')  # oldest start 2015-01-01
g_train_stopDT=datetime.datetime.strptime('2018-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')
g_backtest_stopDT=datetime.datetime.strptime('2018-01-06 09:00:00', '%Y-%m-%d %H:%M:%S')

g_max_step = 20000

#策略参数
g_week=30  #freqency
g_max_holding_days=15


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

#获取板块指数成分股
g_test_securities=gmTools.get_index_stock(STOCK_BLOCK)[:MAX_STOCKS]

'''
    生成共用的大盘数据
    沪深指数统一周期返回k线数据不相同
'''
'''
index_kdata=pd.DataFrame()
for index in INDEX_LIST:
    cols = ['close', 'volume', 'amount']
    if len(index_kdata.columns)==0:
        index_kdata = gmTools.read_kline(index, int(g_week * 60),
                g_train_startDT, g_train_stopDT, 50000)  # 训练数据
        index_kdata.rename(columns={'close':'close1', 'volume':'volume1', 'amount':'amount1'},
                           inplace = True)
    else:
        kdata = gmTools.read_kline(index, int(g_week * 60),
                g_train_startDT, g_train_stopDT, 50000)  # 训练数据

        if len(kdata)==len(index_kdata):
            for col in cols:
                index_kdata[index+col]=kdata[col].values.tolist()


'''

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
        look_back_weeks=g_look_back_weeks,max_count=50000):

    global  g_market_train_data,g_input_columns,\
        g_normalized_data,g_max_step,\
        train_x,train_y,g_current_train_stop

    g_current_train_stop = 0
    #g_market_train_data=index_kdata[:]

    stock_kdata=gmTools.read_kline(stock,int(week*60),
            start_DateTime,stop_DateTime,max_count)    #训练数据
    if len(stock_kdata)==0:
        return
    #预测look_back_weeks周期后的收益
    stock_kdata['label']=stock_kdata['close'].pct_change(look_back_weeks)
    stock_kdata['label']=stock_kdata['label'].shift(-look_back_weeks)
    stock_kdata['label'] = stock_kdata['label'].fillna(0)
    stock_kdata['label'] = stock_kdata['label'].apply(make_stage)

    #将数据总项数整理成g_max_holding_weeks的整数倍
    #tmp = len(g_market_train_data)%g_max_holding_weeks+g_max_holding_weeks

    data_tmp = stock_kdata.iloc[:, 1:-1]
    #todo  加入其他的技术分析指标
    #data_tmp=add_ta_factors(data_tmp)
    # 数据归一化处理
    data_tmp= data_tmp.fillna(0)

    mean = np.mean(data_tmp, axis=0)
    std = np.std(data_tmp, axis=0)
    g_normalized_data = (data_tmp - mean) / std  # 标准化

    g_input_columns=len(data_tmp.columns)

    cols=['eob', 'close','label']
    g_market_train_data = stock_kdata[cols]
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


#基于talib产生每个周期的技术指标因子
def add_ta_factors(kdata):
    opening = kdata['open'].values
    closing = kdata['close'].values
    highest = kdata['high'].values
    lowest = kdata['low'].values
    volume =np.double( kdata['volume'].values)
    tmp = kdata

    # RSI
    tmp['RSI1'] = talib.RSI(closing, timeperiod=6)
    tmp['RSI2'] = talib.RSI(closing, timeperiod=14)
    tmp['RSI3'] = talib.RSI(closing, timeperiod=26)
    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)

    # MACD
    tmp['MACD_DIF'], tmp['MACD_DEA'], tmp['MACD_bar'] = \
        talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)

    '''
    # EMA 指数移动平均线
    tmp['EMA6'] = talib.EMA(closing, timeperiod=6)
    tmp['EMA12'] = talib.EMA(closing, timeperiod=12)
    tmp['EMA26'] = talib.EMA(closing, timeperiod=26)
    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    tmp['OBV'] = talib.OBV(closing, volume)
    
        # 中位数价格 不知道是什么意思
    tmp['MEDPRICE'] = talib.MEDPRICE(highest, lowest)

    # 负向指标 负向运动
    tmp['MiNUS_DI'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['MiNUS_DM'] = talib.MINUS_DM(highest, lowest, timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM'] = talib.MOM(closing, timeperiod=10)

    # 归一化平均值范围
    tmp['NATR'] = talib.NATR(highest, lowest, closing, timeperiod=14)
    # PLUS_DI 更向指示器
    tmp['PLUS_DI'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['PLUS_DM'] = talib.PLUS_DM(highest, lowest, timeperiod=14)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(closing, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROC6'] = talib.ROC(closing, timeperiod=6)
    tmp['ROC20'] = talib.ROC(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROC6'] = talib.ROC(volume, timeperiod=6)
    tmp['VROC20'] = talib.ROC(volume, timeperiod=20)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROCP6'] = talib.ROCP(closing, timeperiod=6)
    tmp['ROCP20'] = talib.ROCP(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROCP6'] = talib.ROCP(volume, timeperiod=6)
    tmp['VROCP20'] = talib.ROCP(volume, timeperiod=20)
'''
    '''
    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    tmp['AD'] = talib.AD(highest, lowest, closing, volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    tmp['ADOSC'] = talib.ADOSC(highest, lowest, closing, volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX'] = talib.ADX(highest, lowest, closing, timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR'] = talib.ADXR(highest, lowest, closing, timeperiod=14)

    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(closing, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，
    # 帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown'], tmp['AROONUp'] = talib.AROON(highest, lowest, timeperiod=14)
    tmp['AROONOSC'] = talib.AROONOSC(highest, lowest, timeperiod=14)


    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR14'] = talib.ATR(highest, lowest, closing, timeperiod=14)
    tmp['ATR6'] = talib.ATR(highest, lowest, closing, timeperiod=6)

    # 布林带
    tmp['Boll_Up'], tmp['Boll_Mid'], tmp['Boll_Down'] = \
        talib.BBANDS(closing, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(opening, highest, lowest, closing)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI5'] = talib.CCI(highest, lowest, closing, timeperiod=5)
    tmp['CCI10'] = talib.CCI(highest, lowest, closing, timeperiod=10)
    tmp['CCI20'] = talib.CCI(highest, lowest, closing, timeperiod=20)
    tmp['CCI88'] = talib.CCI(highest, lowest, closing, timeperiod=88)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如
    # 相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close'] = talib.CMO(closing, timeperiod=14)
    tmp['CMO_Open'] = talib.CMO(opening, timeperiod=14)

    # DEMA双指数移动平均线
    tmp['DEMA6'] = talib.DEMA(closing, timeperiod=6)
    tmp['DEMA12'] = talib.DEMA(closing, timeperiod=12)
    tmp['DEMA26'] = talib.DEMA(closing, timeperiod=26)

    # DX 动向指数
    tmp['DX'] = talib.DX(highest, lowest, closing, timeperiod=14)

    # KAMA 适应性移动平均线
    tmp['KAMA'] = talib.KAMA(closing, timeperiod=30)
    '''


    '''
    # TEMA
    tmp['TEMA6'] = talib.TEMA(closing, timeperiod=6)
    tmp['TEMA12'] = talib.TEMA(closing, timeperiod=12)
    tmp['TEMA26'] = talib.TEMA(closing, timeperiod=26)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(highest, lowest, closing)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(highest, lowest, closing)

    # TSF 时间序列预测
    tmp['TSF'] = talib.TSF(closing, timeperiod=14)

    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(highest, lowest, closing, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR'] = talib.WILLR(highest, lowest, closing, timeperiod=14)
'''
    return tmp

def get_factors(kdata,
                rolling=26,
                drop=False,
                normalization=True):

    opening=kdata['open']
    closing=kdata['close']
    highest=kdata['high']
    lowest=kdata['low']
    volume=kdata['volume']
    tmp = kdata


    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    tmp['AD'] = talib.AD(highest, lowest, closing, volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    tmp['ADOSC'] = talib.ADOSC(highest, lowest, closing, volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX'] = talib.ADX(highest, lowest, closing, timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR'] = talib.ADXR(highest, lowest, closing, timeperiod=14)

    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(closing, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，
    # 帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown'], tmp['AROONUp'] = talib.AROON(highest, lowest, timeperiod=14)
    tmp['AROONOSC'] = talib.AROONOSC(highest, lowest, timeperiod=14)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR14'] = talib.ATR(highest, lowest, closing, timeperiod=14)
    tmp['ATR6'] = talib.ATR(highest, lowest, closing, timeperiod=6)

    # 布林带
    tmp['Boll_Up'], tmp['Boll_Mid'], tmp['Boll_Down'] = \
        talib.BBANDS(closing, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(opening, highest, lowest, closing)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI5'] = talib.CCI(highest, lowest, closing, timeperiod=5)
    tmp['CCI10'] = talib.CCI(highest, lowest, closing, timeperiod=10)
    tmp['CCI20'] = talib.CCI(highest, lowest, closing, timeperiod=20)
    tmp['CCI88'] = talib.CCI(highest, lowest, closing, timeperiod=88)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如
    # 相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close'] = talib.CMO(closing, timeperiod=14)
    tmp['CMO_Open'] = talib.CMO(opening, timeperiod=14)

    # DEMA双指数移动平均线
    tmp['DEMA6'] = talib.DEMA(closing, timeperiod=6)
    tmp['DEMA12'] = talib.DEMA(closing, timeperiod=12)
    tmp['DEMA26'] = talib.DEMA(closing, timeperiod=26)

    # DX 动向指数
    tmp['DX'] = talib.DX(highest, lowest, closing, timeperiod=14)

    # EMA 指数移动平均线
    tmp['EMA6'] = talib.EMA(closing, timeperiod=6)
    tmp['EMA12'] = talib.EMA(closing, timeperiod=12)
    tmp['EMA26'] = talib.EMA(closing, timeperiod=26)

    # KAMA 适应性移动平均线
    tmp['KAMA'] = talib.KAMA(closing, timeperiod=30)

    # MACD
    tmp['MACD_DIF'], tmp['MACD_DEA'], tmp['MACD_bar'] = \
        talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)

    # 中位数价格 不知道是什么意思
    tmp['MEDPRICE'] = talib.MEDPRICE(highest, lowest)

    # 负向指标 负向运动
    tmp['MiNUS_DI'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['MiNUS_DM'] = talib.MINUS_DM(highest, lowest, timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM'] = talib.MOM(closing, timeperiod=10)

    # 归一化平均值范围
    tmp['NATR'] = talib.NATR(highest, lowest, closing, timeperiod=14)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    tmp['OBV'] = talib.OBV(closing, volume)

    # PLUS_DI 更向指示器
    tmp['PLUS_DI'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=14)
    tmp['PLUS_DM'] = talib.PLUS_DM(highest, lowest, timeperiod=14)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(closing, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROC6'] = talib.ROC(closing, timeperiod=6)
    tmp['ROC20'] = talib.ROC(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROC6'] = talib.ROC(volume, timeperiod=6)
    tmp['VROC20'] = talib.ROC(volume, timeperiod=20)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROCP6'] = talib.ROCP(closing, timeperiod=6)
    tmp['ROCP20'] = talib.ROCP(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROCP6'] = talib.ROCP(volume, timeperiod=6)
    tmp['VROCP20'] = talib.ROCP(volume, timeperiod=20)

    # RSI
    tmp['RSI'] = talib.RSI(closing, timeperiod=14)

    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)

    # TEMA
    tmp['TEMA6'] = talib.TEMA(closing, timeperiod=6)
    tmp['TEMA12'] = talib.TEMA(closing, timeperiod=12)
    tmp['TEMA26'] = talib.TEMA(closing, timeperiod=26)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(highest, lowest, closing)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(highest, lowest, closing)

    # TSF 时间序列预测
    tmp['TSF'] = talib.TSF(closing, timeperiod=14)

    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(highest, lowest, closing, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR'] = talib.WILLR(highest, lowest, closing, timeperiod=14)

    # 标准化  todo 不明白用途，暂时保留
    if normalization:
        factors_list = tmp.columns.tolist()[1:]

        if rolling >= 26:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean())\
                         / tmp[i].rolling(window=rolling, center=False).std()
        elif rolling < 26 & rolling > 0:
            print('Recommended rolling range must greater than 26')
        elif rolling <= 0:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].mean()) / tmp[i].std()

    if drop:
        tmp.dropna(inplace=True)
    else:
        tmp=tmp.fillna(0)

    return tmp



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


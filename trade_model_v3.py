# coding=utf-8

import tensorflow as tf
import numpy  as np
import talib
import gmTools_v2 as gmTools
import time
import  matplotlib.pyplot as plt
import pandas as pd
import datetime
import struct

""" 
2018-03-15:
    1)设计成多个神经网络，预测网络解决确定股票池的功能，买卖网络确定最佳买卖时机
      
2018-02-28:
    1)考虑把目前的基于股票的串行训练方式改为全标的按时间段统一训练的平行模式，防止利用了未来信息；
    由于各标的情况不一，同一时段内部分标的可能存在停牌情况，数据项不一定相等，采用在特定时段内按
    标的逐一进行训练的方法，时间分为：训练时间段【2015-01-01 至2016-06-01】，
    回测时间段【2016-06-02 至2017-12-31，暂不考虑中间要空g_max_holding_weeks项数据】，
    实盘时间段【2018年起】；
    2）考虑增加图形显示当前分析股票的价格走势图；
    
2018-02-27:
    1)PyCharm集成开发环境非常耗内存，程序运行时切换到IDLE（Ipython)运行，内存占用显著下降。
    2)由于大数据量运算时占用内存很大，考虑对处理过的数据进行释放，减少内存占用。
    3)增加大盘指数（SHSE.000001 上证综指，SHSE.000002 上证A指，SZSE.399001 深证成指，
      SZSE.399005 中小板指，SZSE.399006 创业板指）、全市场动态数据，评估分析选择沪深300（SZSE.399300）、
     上证50（SHSE.000016）。
    4)同一模型不能在同一次运行中多次加载。 
    
2018-02-25:
    1）在基本功能的基础上，增加基于特定stock的模型管理，图中增加用于记录本模型适应的stock代码、
    训练过的时间范围、频率、模型平均准确率等信息，后续使用时先判断是否有可用的模型，如果有即加载
    模型后继续训练或直接使用；
    2）模型应能持续优化，使用时间越长、精度应该越高；
    3)无法解决utc时间到字符串的转后的结果在gm读取信息中的使用问题，int(week*60)强制类型转换
    4)以股票为单位进行循环训练，数据项超过6000即停止K线数据读取，处理完毕后继续进行；
    
2018-02-22：
    基于本框架进行交易模型设计。首先立足特定股票的波段操作，基本思路：
    1）建立趋势模型：当前时点t的n周期走势划分为11级，对应0，+-1，+-2，+-3，+-4，+-5，
    5表示大于等于5%涨幅，-5小于等于5%跌幅；可扩充为k的倍数，但级数保持11级；【0225已实现】
    1 week的交易数据数组类似一张图片的数据数组；
    2）基本模型：利用ochl，vol，amount；【0225已实现】
      增强模型：上证、深证、沪深300主要大盘指数进行学习与预测,基于level2的
      买卖资金量，talib支持的其他技术指标；
    3）基本模型基于5分钟数据进行学习，模拟盘中利用1分钟数据不断增强模型的适应性；
      数据矩阵[,9] ;【0225已实现】
    4）买卖时机：全市场股票升跌数据统计，用于指导买卖，特例：每次市场大跌是否有征兆？
      
----------------------------------------------------------------------------------
非常清晰明了的介绍，适合学习模仿。
首先载入Tensorflow，并设置训练的最大步数为1000,学习率为0.001,dropout的保留比率为0.9。 
同时，设置MNIST数据下载地址data_dir和汇总数据的日志存放路径log_dir。 
这里的日志路径log_dir非常重要，会存放所有汇总数据供Tensorflow展示。 
"""

#沪深300（SHSE.000300）、SHSE.000947 	内地银行  SHSE.000951 	300银行
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000300'

g_input_columns=6+7
MAX_HOLDING=5
MAX_STOCKS=500

g_train_startDT=datetime.datetime.strptime('2016-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')  # oldest start 2015-01-01
g_train_stopDT=datetime.datetime.strptime('2018-03-01 09:00:00', '%Y-%m-%d %H:%M:%S')
g_backtest_stopDT=datetime.datetime.strptime('2018-04-01 09:00:00', '%Y-%m-%d %H:%M:%S')

BUY_GATE=7
SELL_GATE=3
BUY_FEE=1E-4
SELL_FEE=1E-4
DAY_SECONDS=24*60*60
g_max_step = 20000
g_learning_rate = 0.001
g_dropout = 0.9

#策略参数
g_week=60  #freqency
g_max_holding_days=15


g_trade_minutes=240
g_week_in_trade_day=int(g_trade_minutes/g_week)
g_look_back_weeks=max(10,g_week_in_trade_day*2)*10  #回溯分析的周期数
g_max_holding_weeks=g_week_in_trade_day*g_max_holding_days  #用于生成持仓周期内的收益等级


g_max_stage=11  #持仓周期内收益等级
g_stage_rate=2  if g_week>30 else 1#持仓周期内收益等级差

g_log_dir = '/01deep-ml/logs/w{0}hold{1}days'.format(g_week,g_max_holding_days)


g_current_train_stop=0                  #当前测试数据结束位置
g_test_stop=0                   #当前实时数据结束位置
g_stock_current_price_list=0

train_x=0
train_y=0

g_test_securities=["SZSE.002415","SZSE.000333","SZSE.002460",
                   "SZSE.000001","SZSE.002465","SZSE.002466",
"SZSE.000651","SZSE.000725","SZSE.002152","SZSE.000538","SZSE.300072",
"SHSE.603288","SHSE.600703","SHSE.600271", "SHSE.600690", "SHSE.600585", "SHSE.600271",
"SHSE.600000","SHSE.600519"]

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
        y = data.iloc[start-look_back_weeks:start,2]
        start+=1 #look_back_weeks

        train_x.append(x.values.tolist())

        #test_y.extend(y.values.tolist())
        train_y.append(y.values.tolist())

    return train_x, train_y

def create_market_data(stock,start_DateTime,stop_DateTime,
        week=g_week,look_back_weeks=g_look_back_weeks,hold_weeks=g_max_holding_weeks):

    global  g_market_train_data,g_input_columns,\
        g_normalized_data,g_max_step,train_x,train_y

    g_market_train_data=gmTools.read_kline(stock,int(week*60),
            start_DateTime,stop_DateTime,50000)    #训练数据
    if len(g_market_train_data)==0:
        return
    #预测look_back_weeks周期后的收益
    g_market_train_data['label']=g_market_train_data['close'].pct_change(hold_weeks)
    g_market_train_data['label']=g_market_train_data['label'].shift(-hold_weeks)
    #将数据总项数整理成g_max_holding_weeks的整数倍
    #tmp = len(g_market_train_data)%g_max_holding_weeks+g_max_holding_weeks
    #g_market_train_data =g_market_train_data[tmp:]
    g_market_train_data['label'] =g_market_train_data['label'].fillna(0)
    g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    data_tmp = g_market_train_data.iloc[:, 1:-1]
    #todo  加入其他的技术分析指标

    data_tmp=add_ta_factors(data_tmp)
    # 数据归一化处理
    data_tmp = data_tmp.fillna(0)

    mean = np.mean(data_tmp, axis=0)
    std = np.std(data_tmp, axis=0)
    g_normalized_data = (data_tmp - mean) / std  # 标准化

    g_input_columns=len(data_tmp.columns)

    cols=['eob', 'close','label','volume', 'amount']  #买卖点分析需要量价信息
    g_market_train_data = g_market_train_data[cols]
    g_max_step = len(g_market_train_data)

    #数据规整为look_back_weeks的整数倍
    #remainder=len(g_market_train_data)%look_back_weeks
    #g_market_train_data=g_market_train_data[remainder:]
    #g_normalized_data = g_normalized_data[remainder:]

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


# 定义对Variable变量的数据汇总函数
""" 
计算出var的mean,stddev,max和min， 
对这些标量数据使用tf.summary.scalar进行记录和汇总。 
同时，使用tf.summary.histogram直接记录变量var的直方图。 
"""
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# 定义神经网络模型参数的初始化方法，
# 权重依然使用常用的truncated_normal进行初始化，偏置则赋值为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 设计一个MLP多层神经网络来训练数据，在每一层中都会对模型参数进行数据汇总。
""" 
定一个创建一层神经网络并进行数据汇总的函数nn_layer。 
这个函数的输入参数有输入数据input_tensor,输入的维度input_dim,
输出的维度output_dim和层名称layer_name，激活函数act则默认使用Relu。 
在函数内，显示初始化这层神经网络的权重和偏置，并使用前面定义的
variable_summaries对variable进行数据汇总。 
然后对输入做矩阵乘法并加上偏置，再将未进行激活的结果使用tf.summary.histogram统计直方图。 
同时，在使用激活函数后，再使用tf.summary.histogram统计一次。 
"""
def nn_layer(input_tensor, input_dim,
             output_dim, layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            #variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases)

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)

        activations = act(preactivate, name='actvations')
        tf.summary.histogram('activations', activations)
        return activations

""" 
使用刚定义好的nn_layer创建一层神经网络，输入维度（1*g_input_columns），
输出的维度是隐藏节点数500. 
再创建一个Droput层，并使用tf.summary.scalar记录keep_prob。
然后再使用nn_layer定义神经网络的输出层，激活函数为全等映射，
此层暂时不使用softmax,在后面会处理。 
"""

g_nn_hidden_nodes=0
hidden1=0
x=0
y=0
y1=0
cross_entropy=0
train_step=0
buy=0
sell=0
accuracy=0
merged=0
train_writer=0
test_writer=0
keep_prob=0
sell_prediction=0
buy_prediction=0
valid_accuracy=0
valid_accuracy2=0
model_code=0
model_last_utc=0
model_next_train_utc=0
model_week=0
reward_prediction=0

def setup_tensor(sess,stock,week,last_utc,next_train_time=0):
    global  g_nn_hidden_nodes,hidden1,cross_entropy,\
        train_step,buy,sell,accuracy,merged,train_writer,\
        test_writer,keep_prob,x,y,y1,sell_prediction,\
        buy_prediction,valid_accuracy,valid_accuracy2,\
        model_code,model_last_utc,model_week,\
        model_next_train_utc,reward_prediction

    """ 
    为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间， 
    在这个with下所有的节点都会自动命名为input/xxx这样的格式。 
    定义输入x和y的placeholder，并将输入的一维数据变形为28×28的图片存储到另一个tensor， 
    这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示了。 
    """

    #定义网络参数
    with tf.name_scope("model_vars"):
        model_code=tf.Variable(stock,dtype=tf.string,trainable=False,name="model_code")
        model_week = tf.Variable(week, dtype=tf.int32, trainable=False, name="model_week")
        model_last_utc=tf.Variable(last_utc, dtype=tf.int64,
                    trainable=False, name="model_last_utc")
        model_next_train_utc = tf.Variable(next_train_time, dtype=tf.int64,
                                     trainable=False, name="model_next_train_utc")

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
            1 * g_input_columns], name='x_input')
        y = tf.placeholder(tf.float32, [None, g_max_stage], name='y_input')

    g_nn_hidden_nodes=800
    hidden1 = nn_layer(x, 1*g_input_columns, g_nn_hidden_nodes, 'layer1')

    with tf.name_scope('g_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        #tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y1 = nn_layer(dropped, g_nn_hidden_nodes, g_max_stage, 'layer2', act=tf.identity)

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
        #ones= tf.constant(1.0, shape=[g_max_stage])
        #loss=tf.reduce_sum(1-accuracy)

        reward_prediction = tf.argmax(y1, 1)

        # 误差在1个单位增益范围内的结果
        valid_prediction = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 1)
        error=tf.reduce_mean(tf.cast(tf.abs( tf.argmax(y1, 1)-tf.argmax(y, 1)),tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
        tf.summary.scalar('valid_accuracy', valid_accuracy)
        tf.summary.scalar('error', error)

        # 误差在两个单位增益范围内的结果
        valid_prediction2 = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 2 )
        valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))
        tf.summary.scalar('valid_accuracy2', valid_accuracy2)


        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
        #diff = tf.nn.softmax_cross_entropy_with_logits(logits=prediction_mean, labels=true_mean)
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    """ 
    使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray， 
    再使用tf.summary.scalar对accuracy进行统计汇总。 
    train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)
    """
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)

    """ 
    由于之前定义了非常多的tf.summary的汇总操作，一一执行这些操作态麻烦， 
    所以这里使用tf.summary.merger_all()直接获取所有汇总操作，以便后面执行。 
    然后，定义两个tf.summary.FileWrite(文件记录器)在不同的子目录，
    分别用来存放训练和测试的日志数据。 
    同时，将Session的计算图sess.graph加入训练过程的记录器，
    这样在TensorBoard的GRAPHS窗口中就能展示整个计算图的可视化效果。 
    最后使用tf.global_variables_initializer().run()初始化全部变量。 
    """

    merged = tf.summary.merge_all()
    #自动生成工程需要的文件目录
    #train_writer = tf.summary.FileWriter(g_log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(g_log_dir + '/test')
    test_figure = tf.summary.FileWriter(g_log_dir + '/fig')
    tf.global_variables_initializer().run()


def feed_dict(train):
    global  g_current_train_stop

    xs = train_x[g_current_train_stop]
    ys = train_y[g_current_train_stop]

    #todo  数据取完后如何处理？  直接退出运行,需支持对未来收益的预测
    if train:
        k = g_dropout
        g_current_train_stop += 1
    else:
        k = 1.0

    return {x: xs, y: ys, keep_prob: k}


# 实际执行具体的训练，测试及日志记录的操作
""" 
首先，使用tf.train.Saver()创建模型的保存器。 
然后，进入训练的循环中，每隔10步执行一次merged（数据汇总），
accuracy（求测试集上的预测准确率）操作， 
并使应test_write.add_summary将汇总结果summary和循环步数i写入日志文件; 
同时每隔100步，使用tf.RunOption定义Tensorflow运行选项，其中设置trace_level为FULL——TRACE, 
并使用tf.RunMetadata()定义Tensorflow运行的元信息， 
这样可以记录训练是运算时间和内存占用等方面的信息. 
再执行merged数据汇总操作和train_step训练操作，将汇总summary和训练元信息run_metadata添加到train_writer. 
平时，则执行merged操作和train_step操作，并添加summary到trian_writer。 
所有训练全部结束后，关闭train_writer和test_writer。 
"""
def train_model( week=g_week,look_back_weeks=g_look_back_weeks):
    global  g_train_startDT,g_current_train_stop,g_market_train_data,\
        g_normalized_data,g_max_step,train_x,train_y

    sess = tf.InteractiveSession()
    setup_tensor(sess, STOCK_BLOCK, g_week, 0)
    saver = tf.train.Saver(max_to_keep=len(g_test_securities)+2)

    sess.run(tf.assign(model_week, g_week))

    ii=0
    total_count=0
    last_acc1=-10
    last_acc=-10
    no_change_count=0

    for stock in g_test_securities:
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
                ii*100/len(g_test_securities)))


        create_market_data(stock=stock,
                           start_DateTime=g_train_startDT,
                           stop_DateTime=g_train_stopDT ,
                           week=week, look_back_weeks=g_look_back_weeks)

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
            feed_dict_data=feed_dict(True)
            #
            summary,_= sess.run([merged,train_step],feed_dict=feed_dict_data)

            if total_count% 20==0:
                train_writer.add_summary(summary, i)
                #tmp=sess.run(cross_entropy,feed_dict=feed_dict_data)

            total_count += 1

        #评估训练的效能  同一模型不能打开两次
        '''
        if ii/len(g_test_securities)>0.3:
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
        g_normalized_data = 0
        g_max_step = 0
        train_x = 0
        train_y = 0


        sess.run(tf.assign(model_code,stock))
        saver.save(sess, model_path + stock+'/model.ckpt')  # save模型  for stock

        sess.run(tf.assign(model_code, STOCK_BLOCK))
        saver.save(sess, model_path +STOCK_BLOCK+ '.BLOCK/model.ckpt')  # save模型  for all

    sess.close()
    print('total train %d steps'%(total_count))

'''
    在价格走势图显示买卖点信息
'''
def draw_bs_on_kline(stock,kdata,buy_time,sell_time,week=g_week):
    # 以折线图表示结果 figsize=(20, 15)
    try:
        plt.figure(figsize=(16, 9))
        closing = kdata['close'].values
        data=closing.tolist()

        plt.plot(list(range(len(kdata))),
                    data, color='b', label='close')

        # EMA 指数移动平均线  ma6跌破ma12连续若干周期
        ma6 = talib.EMA(closing, timeperiod=6)
        ma12 = talib.EMA(closing, timeperiod=12)
        ma26 = talib.EMA(closing, timeperiod=26)

        plt.plot(list(range(len(kdata))),
                 ma6,color='r',label='ma6')
        plt.plot(list(range(len(kdata))),
                 ma12, color='y',label = 'ma12')
        plt.plot(list(range(len(kdata))),
                 ma26, color='g', label='ma26')

        time_list=kdata['eob'].tolist()

        x = time_list.index(buy_time)
        buy_price = data[x]
        
        if buy_time==sell_time:
            #no sell time,sell on maxholding weeks
            sell_time=time_list[x+g_max_holding_weeks]
            no_sell=True
        else:
            no_sell=False

        plt.annotate(str(x), xy=(x, buy_price),
                     xytext=(x * 1.1, buy_price),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     )
        x = time_list.index(sell_time)
        sell_price = data[x]

        plt.annotate(str(x), xy=(x, sell_price),
                     xytext=(x * 0.9, sell_price),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     )
        #display start date,mid date and stop date
        x=5
        date_high=min(closing)
        plt.annotate(str(time_list[x].date()), xy=(x, date_high),
                     xytext=(x , date_high),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )

        x =int(len(time_list)/2)
        plt.annotate(str(time_list[x].date()), xy=(x, date_high),
                     xytext=(x, date_high),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )

        x = len(time_list)-5
        plt.annotate(str(time_list[x].date()), xy=(x, date_high),
                     xytext=(x , date_high),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )

        reward=int(sell_price*100/buy_price-100)
        buy_time = buy_time.strftime('%Y-%m-%d %H-%M-%S')
        sell_time = sell_time.strftime('%Y-%m-%d %H-%M-%S')
        title = '%s week=%d reward=%d%%\n %s--%s'%(stock, week,reward,buy_time,sell_time)

        if no_sell:
            title='no sell ' +title

        plt.title(title)
    except:
        pass



    plt.legend(loc='upper left', shadow=True, fontsize='x-large')


    file = '%s/%03d-%s-%s-%s.png'%(
        g_log_dir + '/fig',reward,stock, buy_time, sell_time )

    plt.savefig(file)
    plt.close()

    return  reward

#判断当前走势能否买入？
#todo use tf model to decice the buy-sell point
def can_buy(kdata,week=3):
    ret=False
    closing = kdata['close'].values

    while not ret:
        # RSI
        RSI1 = talib.RSI(closing, timeperiod=6)[-1]
        RSI2 = talib.RSI(closing, timeperiod=14)[-1]
        RSI3 = talib.RSI(closing, timeperiod=26)[-1]
        #RSI1>RSI2 and RSI2>RSI3
        if  (  RSI1<40 or RSI1>80) :
            break

        # MACD  bar 最新三项大于0且处于金叉后的上升阶段
        dif, dea, bar = talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)
        bar = bar[-week:]
        dif = dif[-week:]
        dea = dea[-week:]

        for i in range(1, week):
            if bar[i] < bar[i - 1] \
                or dif[i] < dif[i - 1] \
                or dea[i] < dea[i - 1]:
                break

        if i < week - 1:
            break

        # EMA 指数移动平均线  多头发散
        ma6= talib.EMA(closing, timeperiod=6)[-week:]
        ma12 = talib.EMA(closing, timeperiod=12)[-week:]
        ma26 = talib.EMA(closing, timeperiod=26)[-week:]

        for i in range(week):
            if ma6[-i]<ma12[-i]  \
               or ma12[-i]<ma26[-i]:
                break

        if i<week-1:
            break

        ret=True

    return  ret

def can_sell(kdata,week=3):
    ret=False
    closing = kdata['close'].values

    while not ret:
        # EMA 指数移动平均线  ma6跌破ma12连续若干周期
        ma6= talib.EMA(closing, timeperiod=6)[-week:]
        ma12 = talib.EMA(closing, timeperiod=12)[-week:]
        #ma26 = talib.EMA(closing, timeperiod=26)
        for i in range(week):
            if ma6[-i]>=ma12[-i] :
                break

        if i<week-1:
            break

        # MACD  bar 最新三项大处于逐步减少的阶段，卖出
        dif, dea, bar = talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)
        bar=bar[-week:]
        for i in range(1,week):
            if bar[-i]>=bar[-i+1] :
                break

        if i<week-1:
            break

        ret=True

    return  ret
# 基于talib产生每个周期的技术指标因子
def add_ta_factors(kdata):
    opening = kdata['open'].values
    closing = kdata['close'].values
    highest = kdata['high'].values
    lowest = kdata['low'].values
    #volume = np.double(kdata['volume'].values)
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



'''
    由于掘金返回批量股票K线数据很慢，采用集中获取买卖点信息后统一处理，
    自行计算买卖数据，自行统计盈利情况
'''
def backtest_model( week=g_week,look_back_weeks=g_look_back_weeks,
                    startDT =g_train_stopDT,stopDT=g_backtest_stopDT):

    print("\nstart backtest_model [%s]" % (
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    ii=0

    sess = tf.InteractiveSession()
    setup_tensor(sess, '', g_week, 0)
    saver = tf.train.Saver()

    acc_list=[]
    acc, acc1, total=0,0,0

    bacttest_start_date=gmTools.get_backtest_start_date(g_train_stopDT,
                    int(1.5*g_look_back_weeks/g_week_in_trade_day))

    for stock in g_test_securities:
        try:
            print("\n[%s]start backtesting [%s] %.2f%%" % (stock,
                   time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   ii * 100 / len(g_test_securities)))
            ii += 1
            acc0,acc1,acc2,total=backtest_stock(stock=stock, sess=sess,saver=saver,
                           week=week,look_back_weeks=look_back_weeks,
                           startDT =bacttest_start_date,
                           stopDT=stopDT)

            acc_list.append([stock,acc0,acc1,acc2,total])
        except:
            print('error in backtest_model, doing [%s]'%(stock))

    acc0_total=0
    acc1_total=0
    acc2_total = 0
    totals=0
    for (stock,acc0 , acc1 ,acc2, total) in acc_list:
        if total>0:
            print("[%s] acc0=%.2f%%,acc1=%.2f%%,acc2=%.2f%%" % (
                stock, acc0 * 100 / total, acc1 * 100 / total, acc2 * 100 / total))

        acc0_total+=acc0
        acc1_total += acc1
        acc2_total += acc2
        totals+=total

    if totals>0:
        print("BLOCK  [%s] total acc0=%.2f%%,acc1=%.2f%%,acc2=%.2f%%" % (STOCK_BLOCK,
                acc0_total * 100 / totals, acc1_total * 100 / totals, acc2_total * 100 / totals))

    sess.close()

    return acc,acc1,total

'''
    基于特定股票的回测
'''
def backtest_stock(stock,sess,saver, week=g_week,
       look_back_weeks=g_look_back_weeks,hold_weeks=g_max_holding_weeks,
       startDT =g_train_stopDT,stopDT=g_backtest_stopDT):

    global g_train_startDT, g_current_train_stop, g_market_train_data, \
        g_normalized_data, g_max_step, train_x, train_y,reward_prediction

    acc0 = 0
    acc1 = 0
    acc2 = 0
    total = 0

    if restore_stock_model(stock,sess,saver)==False:
        return acc, acc1, total

    create_market_data(stock=stock,
                       start_DateTime=startDT ,
                       stop_DateTime=stopDT ,
                       week=week, look_back_weeks=g_look_back_weeks)

    g_max_step = len(g_market_train_data)
    if g_max_step < look_back_weeks:
        g_market_train_data=0
        return acc, acc1, total


    train_count =g_max_step  # int(g_max_step / look_back_weeks)
    print('log dir:%s , total items :%d' % (g_log_dir, g_max_step))

    g_current_train_stop = 0
    i = g_current_train_stop
    try:
        print("backtesting %s" % (
            g_market_train_data.iloc[0, 0].strftime('%Y-%m-%d %H:%M:%S')))
    except:
        print('error backtesting %s ' % (stock))
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
        if g_market_train_data.iloc[i, 0]<g_train_stopDT:
            feed_dict(True)
            continue


        feed_dict_data = feed_dict(False)
        reward,acc,valid_acc,valid_acc1 = sess.run(
            [reward_prediction,accuracy,valid_accuracy,valid_accuracy2],
            feed_dict=feed_dict_data)

        if acc>0.5:
            acc0+=1

        if valid_acc>0.5:
            acc1+=1

        if valid_acc1>0.5:
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
            if not buy_on and last_reward[index - 1] >= BUY_GATE \
               and last_reward[index - 2] >= BUY_GATE:
                #利用当前的价格走势确定是否为有效买点
                #todo maup,rsi,macd??  use tf model to make decision
                kdata=g_market_train_data.iloc[ i- look_back_weeks:i, :]

                if can_buy(kdata):
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

                if can_sell(kdata) \
                  or i  - buy_point > g_max_holding_weeks:  # arrive max holding weeks

                    buy_on = False
                    code_4_sell.append(
                        {'code': stock,
                        'time': g_market_train_data.iloc[i, 0],
                        'price': g_market_train_data.iloc[ i, 1],
                        'reward': last_reward[index - 1]})

        last_reward = last_reward[-2:]

        # train the model
        feed_dict_data = feed_dict(True)
        summary, _, = sess.run(
            [merged, train_step],
            feed_dict=feed_dict_data)
        # train_writer.add_summary(summary, i)

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
                        vol = int(money / (item['price'] * (1 + BUY_FEE) * 100)) * 100
                        if vol > 0:
                            tmp = vol * item['price'] * (1 + BUY_FEE)
                            amount -= tmp
                            holdings.append(item['code'])

                            start_datetime = bs_time \
                                +datetime.timedelta(days=- g_max_holding_days*2)

                            must_sell_datetime = bs_time \
                                +datetime.timedelta(days=g_max_holding_days * 4)

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
                        tmp = vol * item['price'] * (1 - SELL_FEE)
                        amount += tmp
                        # todo save bs point in the graph
                        draw_bs_on_kline(holding[i]['code'],
                                holding[i]['kdata'], holding[i]['time'], bs_time)

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
        draw_bs_on_kline(holding[i]['code'],
                         holding[i]['kdata'], holding[i]['time'], holding[i]['time'])

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

    print("\nstop backtest_model [%s]" % (
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    return acc0,acc1,acc2,total
'''
    基于特定股票的回测模型恢复
'''
def restore_stock_model(stock,sess,saver):
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

#返回字典列表
def get_bs_list( stop_dt='',week=g_week,
    look_back_weeks=g_look_back_weeks,count=g_look_back_weeks):
    global g_train_startDT, g_current_train_stop, g_market_train_data, \
        g_normalized_data, g_max_step, train_x, train_y

    now = '[{0}] create_market_last_n_data'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(now)

    create_market_last_n_data(g_test_securities,max(count,look_back_weeks),
        stop_dt,week,look_back_weeks)

    code_4_buy=[]
    code_4_sell=[]
    g_max_step = 0
    g_current_train_stop = 0

    for item in g_stock_current_price_list:
        stock=item['code']
        #print("\n[%s]start analysing [%s]" % (stock,
        #    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

        feed_dict_data = feed_dict(False)
        reward = sess.run( reward_prediction, feed_dict=feed_dict_data)

        #todo 简单判断未来走势未必合理，是否考虑看均线趋势？
        if  reward[-1]>=BUY_GATE and reward[-2]>=BUY_GATE:
            code_4_buy.append({'code':stock,
                    'price':item['time_close'][0][1],'reward':reward[-1]})

        if (reward[-1]<=SELL_GATE and reward[-2]<=SELL_GATE):
            code_4_sell.append({'code':stock,
                    'price':item['time_close'][0][1],'reward':reward[-1]})

    g_market_train_data = 0
    g_normalized_data = 0
    g_max_step = 0
    train_x = 0
    train_y = 0

    #sess.close()

    if len(code_4_buy)>0:
        #字典列表按键值‘a’逆序排序  a.sort(key=lambda x:-x['a']
        code_4_buy.sort(key=lambda x:x['reward'],reverse=True)

        #取涨幅最大的前五
        buy_list=code_4_buy[:MAX_HOLDING]
        now='[{0}]'.format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print(now,stop_dt,' buy list:', {x['code'] for x in buy_list})
    else:
        buy_list=[]

    if len(code_4_sell)>0:
        code_4_sell.sort(key=lambda x: x['reward'], reverse=False)
        sell_list = code_4_sell
        now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(now,stop_dt,' sell list:', {x['code'] for x in sell_list[:6]})
    else:
        sell_list=[]

    if len(code_4_sell)==0 and len(code_4_buy)==0:
        now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(now,stop_dt, ' no trade')

    return   buy_list,sell_list

def get_stock_list():
    return g_test_securities

g_test_securities=gmTools.get_index_stock(STOCK_BLOCK)[:MAX_STOCKS]
#study()
if __name__ == '__main__':
    #训练模型
    train_model()
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
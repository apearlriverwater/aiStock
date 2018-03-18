# coding=utf-8

import tensorflow as tf

learning_rate = 0.001
dropout = 0.9

# 定义对Variable变量的数据汇总函数
""" 
计算出var的mean,stddev,max和min， 
对这些标量数据使用tf.summary.scalar进行记录和汇总。 
同时，使用tf.summary.histogram直接记录变量var的直方图。 
"""
def variable_summaries(var,debug=False):
    if debug:
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
             output_dim, layer_name,act=tf.nn.relu,
             debug=False):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights,debug)

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases,debug)

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            if debug:
                tf.summary.histogram('pre_activations', preactivate)

        activations = act(preactivate, name='actvations')
        if debug:
            tf.summary.histogram('activations', activations)

        return activations






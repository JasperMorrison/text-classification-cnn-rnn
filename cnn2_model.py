# coding: utf-8

import tensorflow as tf
from data.app_history_loader import build_pkg_id
from data.app_history_loader import history_num
from data.app_history_loader import other_num

class TCNNConfig(object):
    """CNN配置参数"""

    def __init__(self, train=False):
        self.train = train
    
    train = False
    
    embedding_dim = 150  # 词向量维度，根据应用个数自动生成
    seq_length = 7  # 序列长度，多余的自动填0，input_x的长度，根据history查询输出决定
    num_classes = 99  # 类别数,app的个数,根据应用个数自动生成
    num_filters = 50  # 卷积核数目
    kernel_size = 3  # 卷积核尺寸
    vocab_size = 100  # 最大的可能数值,根据应用个数自动生成

    hidden_dim = 128/20 * num_classes  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 100  # 每批训练大小
    num_epochs = 30  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    in_top_num = history_num / 2 if 1 <= history_num / 2 <= 3 else 1 # 取概率最高的前k个预测结果


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        
        self.config.num_classes = len(build_pkg_id(self.config.train))
        self.config.vocab_size = 24 + self.config.num_classes +1
        self.config.embedding_dim = self.config.vocab_size
        self.config.hidden_dim = 128/10 * self.config.num_classes
        self.config.seq_length = history_num + other_num
        
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            # 对卷积结果取值，结果矩阵取每一行的最大值，组合成一个新的向量
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # 对所有的卷积结果转换成隐藏层的维度
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            # 抛弃一部分
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # relu层，线性整流，max(0, w^T * x + b)
            fc = tf.nn.relu(fc)

            # 分类器
            # 将前一层输出转换成逻辑分类
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # 将分类结果转换成概率，并取概率最大的下标，下标值对应分类结果
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            #self.y_pred_cls = tf.nn.softmax(self.logits)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            y = tf.argmax(self.input_y, 1)
            correct_pred = tf.nn.in_top_k(predictions=self.logits, targets=y, k=self.config.in_top_num)
            #correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

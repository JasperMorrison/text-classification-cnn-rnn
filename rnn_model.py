#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度，一个句子最多600个词
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表大小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元，隐藏层维度。一个神经元对应一个词的输入，对应一个维度
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小，可以任意选择，不会影响最终的训练结果，但会影响训练时长
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        # run_rnn中将赋值为dropout_keep_prob = 0.8
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核，只需要指定核的维度；state_is_tuple=true表示输入、输出、cell的维度相同，都是 batch_size * num_units
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            # 可以理解为：把dropout层添加到cell层后面，返回cell层的引用
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            # 多层带有dropout层的cell层
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            # 动态rnn节省了padding与反padding操作
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果，[batch_size * 1, seq_length * 1, cell_dim * 1]，取最后一个cell的输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # dense() 新建一个全连接层
            # dense()自动将[batch_size * 1, seq_length * 1, cell_dim * 1] 转换为 [batch_size * 1, seq_length * 1, hidden_dim * 1]
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # dropout全连接层，作用是根据比例自动将部分清零
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # relu层，线性整流，max(0, w^T * x + b)
            fc = tf.nn.relu(fc)

            # 分类器
            # dense() 新建一个分类器层，只有经过整流之后，才能进行分类。dense需要一个整流后的输入和一个分类列表，输出对应的x在每一个类别y上的概率值
            # 比如：[0.9, 0.05, 0.03, 0.02, 0, 0]
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # 取最大值，只取最大的概率作为最后的分类结果
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵，我们的训练目标就是希望将loss降到最低
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # 取损失平均值
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器，learning_rate表示梯度下降算法的步子大小，这里是固定的吗？还是内部会自动变化？
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            # 对比预测与输入是否相同，相同返回True，否则返回False
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            # 将True和False转换为Float32，也就是1.0或者0.0
            # 求所有1.0 和 0.0 的平均值，平均值越大（越接近1），表示准确性越高
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

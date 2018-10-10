# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn2_model import TCNNConfig, TextCNN
from data.app_history_loader import read_file, history_num, get_ids_pkgs_pair

try:
    bool(type(unicode))
except NameError:
    unicode = str

predict_file = 'data/app_info/predict.txt'
    
base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/appcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self):
        queue = [None]*history_num
        self.content, label = read_file(predict_file, queue, self.config.num_classes)
        data = self.content[-1]
        print(data)

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        logits = self.session.run(self.model.logits, feed_dict=feed_dict)
        s = tf.nn.softmax(logits)
        b = tf.nn.top_k(s, 5)
        o = self.session.run(b)
        indices = o.indices[0]
        values = o.values[0]
        
        ids = get_ids_pkgs_pair()
        pkgs = []
        for i in indices:
            pkgs.append(ids[i])
        
        print(zip(indices, values))
        return zip(pkgs, values)


if __name__ == '__main__':
    cnn_model = CnnModel()
    print(cnn_model.predict())

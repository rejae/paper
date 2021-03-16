#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
#from interaction_layer import bilinear_attention, cross_attention
from tensorflow.keras import layers
from tensorflow import keras


class TBiRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    word_embed_dim = 150  # 词向量维度 100
    word_seq_length = 50  # 词序列长度 80
    word_vocab_size = 10000  # 词汇表大小

    phoneme_embed_dim = 150  # 音素向量维度 100
    phoneme_seq_length = 50  # 音素序列长度 100
    phoneme_vocab_size = 100  # 音素词表大小

    num_classes = 10  # 类别数
    num_layers = 1  # 隐藏层层数
    hidden_dim = 150  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 0.001  # 学习率
    decay_steps = 1000
    decay_rate = 0.1

    batch_size = 64  # 每批训练大小
    num_epochs = 20  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    operate = 'word_phoneme_add'  # 1.word_single 2.phoneme_single 3.word+phoneme concat
    rg = 0.0

    use_w2vec = False


class TextBiRNN(object):
    """文本分类，RNN模型"""

    def __init__(self, FLAGS,sequence_length,num_classes,vocab_size,embedding_size, embeddings,hidden_dim, e1,e2,e3):

        self.FLAGS=FLAGS
        # 三个待输入的数据
        self.num_classes=num_classes
        self.input_x_trs = tf.placeholder(tf.int32, [None, sequence_length], name='input_x_trs')
        #self.input_x_word_len = tf.placeholder(tf.int32, [None], name='input_x_word_len')  # for mask

        self.input_x_asr = tf.placeholder(tf.int32, [None, sequence_length], name='input_x_asr')
        #self.input_x_phoneme_len = tf.placeholder(tf.int32, [None], name='input_x_phoneme_len')  # for mask

        self.input_y = tf.placeholder(tf.int32, [None,num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.vocab_size = vocab_size
        self.word_embeddings = embeddings
        self.embedding_size=embedding_size
        self.hidden_dim=hidden_dim
        self.e1=e1
        self.e2=e2
        self.e3=e3
        self.birnn()

    def birnn(self):
        """rnn模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            if self.word_embeddings != None:
                # pre-trained embedding for word, and random init embedding for phoneme.
                word_embeddings = tf.Variable(name='word_embeddings', initial_value=self.word_embeddings,
                                              trainable=False)
                '''phone_embeddings = tf.Variable(name='phoneme_embeddings',
                                               initial_value=tf.random_uniform(
                                                   [self.config.phoneme_vocab_size, self.config.phoneme_embed_dim],
                                                   -1.0, 1.0), trainable=True)'''
            else:
                word_embeddings = tf.Variable(name='word_embeddings',
                                              initial_value=tf.random_uniform(
                                                  [self.vocab_size, self.embedding_size],
                                                  -1.0, 1.0), trainable=True)

            # init for input word and phoneme
            embedding_inputs_x_trs = tf.nn.embedding_lookup(word_embeddings, self.input_x_trs)
            embedding_inputs_x_asr = tf.nn.embedding_lookup(word_embeddings, self.input_x_asr)
            #embedding_inputs_x_phoneme = tf.nn.embedding_lookup(phone_embeddings, self.input_x_phoneme)

        # word sequence for bi_rnn
        with tf.name_scope("word_rnn"):
            model_layers=layers.LSTM(self.hidden_dim, return_sequences=True,name='represent_layer')
            trs_outputs = layers.Bidirectional(model_layers)(embedding_inputs_x_trs)
            print("trs_outputs:{}".format(trs_outputs))
            
            asr_outputs = layers.Bidirectional(model_layers)(embedding_inputs_x_asr)
            print("asr_outputs:{}".format(asr_outputs))
            
        final_feat_trs = tf.reduce_mean(trs_outputs, axis=1, name='final_feat_trs')
        final_feat_asr = tf.reduce_mean(asr_outputs, axis=1, name='final_feat_asr')
        '''# phoneme sequence for bi_rnn
        with tf.name_scope("phoneme_rnn"):
            phoneme_outputs = layers.Bidirectional(layers.GRU(self.config.hidden_dim, return_sequences=True))(
                embedding_inputs_x_phoneme)
            print("phoneme_outputs:{}".format(phoneme_outputs))
        # word_outputs: b * w_l * h; phoneme_outputs: b * p_l * h
        if self.config.operate == 'word_single_last_h':
            print("word_last_h", word_last_h)
            final_feat = word_last_h[0]
        elif self.config.operate == 'word_single':
            final_feat = tf.reduce_mean(word_outputs, axis=1, name='final_feat')  # b * h
        elif self.config.operate == 'phoneme_single':
            final_feat = tf.reduce_mean(phoneme_outputs, axis=1, name='final_feat')
        elif self.config.operate == 'word_phoneme_concat':
            word_feat = tf.reduce_mean(word_outputs, axis=1, name='word_feat')
            phoneme_feat = tf.reduce_mean(phoneme_outputs, axis=1, name='phoneme_feat')
            final_feat = tf.concat([word_feat, phoneme_feat],
                                   axis=-1, name='final_feat')  # b, 2 * h
        elif self.config.operate == 'word_phoneme_add':
            word_feat = tf.reduce_mean(word_outputs, axis=1, name='word_feat')  # b * h
            phoneme_feat = tf.reduce_mean(phoneme_outputs, axis=1, name='phoneme_feat')  # b * h

            word_feat = tf.expand_dims(word_feat, 1)  # b * h * 1
            phoneme_feat = tf.expand_dims(phoneme_feat, 1)  # b * h * 1

            final_feat = tf.reduce_mean(tf.concat([word_feat, phoneme_feat], axis=1), axis=1,
                                        name='final_feat')
        elif self.config.operate == 'ban2word':
            # generate mask (1.0 or 0.0) matrix
            input_word_mask = tf.sequence_mask(self.input_x_word_len, maxlen=self.config.word_seq_length,
                                               dtype=tf.float32)
            input_phoneme_mask = tf.sequence_mask(self.input_x_phoneme_len, maxlen=self.config.phoneme_seq_length,
                                                  dtype=tf.float32)

            # word_feature  # b * phoneme_len * h
            # phoneme_feature  # b * phoneme_len * h
            ban_outputs = bilinear_attention(word_outputs, phoneme_outputs, self.config.hidden_dim, input_word_mask,
                                             input_phoneme_mask)  # b * h
            ban_outputs = tf.expand_dims(ban_outputs, axis=1)  # b * 1 * h

            final_feat = ban_outputs + word_outputs  # b * w_s * h
            final_feat = tf.reduce_mean(final_feat, axis=1, name='final_feat')  # b, h
        elif self.config.operate == 'ban2phoneme':
            input_word_mask = tf.sequence_mask(self.input_x_word_len, maxlen=self.config.word_seq_length,
                                               dtype=tf.float32)
            input_phoneme_mask = tf.sequence_mask(self.input_x_phoneme_len, maxlen=self.config.phoneme_seq_length,
                                                  dtype=tf.float32)
            ban_outputs = bilinear_attention(word_outputs, phoneme_outputs, self.config.hidden_dim, input_word_mask,
                                             input_phoneme_mask)  # b * h
            ban_outputs = tf.expand_dims(ban_outputs, axis=1)  # b * 1 * h

            final_feat = ban_outputs + phoneme_outputs  # b * p_s * h
            final_feat = tf.reduce_mean(final_feat, axis=1, name='final_feat')  # b, h
        elif self.config.operate == 'ban2word_phoneme_concat':
            input_word_mask = tf.sequence_mask(self.input_x_word_len, maxlen=self.config.word_seq_length,
                                               dtype=tf.float32)
            input_phoneme_mask = tf.sequence_mask(self.input_x_phoneme_len, maxlen=self.config.phoneme_seq_length,
                                                  dtype=tf.float32)
            ban_outputs = bilinear_attention(word_outputs, phoneme_outputs, self.config.hidden_dim, input_word_mask,
                                             input_phoneme_mask)  # b * h
            ban_outputs = tf.expand_dims(ban_outputs, axis=1)  # b * 1 * h

            ban_phoneme_feature = ban_outputs + phoneme_outputs  # b * p_s * h
            ban_phoneme_feature = tf.reduce_mean(ban_phoneme_feature, axis=1)
            ban_word_feature = ban_outputs + word_outputs  # b * w_s * h
            ban_word_feature = tf.reduce_mean(ban_word_feature, axis=1)
            final_feat = tf.concat([ban_word_feature, ban_phoneme_feature], axis=1, name='final_feat')  # b, 2h
        elif self.config.operate == 'word_phoneme_coatt':
            input_word_mask = tf.sequence_mask(self.input_x_word_len, maxlen=self.config.word_seq_length,
                                               dtype=tf.float32)
            input_phoneme_mask = tf.sequence_mask(self.input_x_phoneme_len, maxlen=self.config.phoneme_seq_length,
                                                  dtype=tf.float32)
            final_feat = cross_attention(word_outputs, phoneme_outputs, input_word_mask, input_phoneme_mask,
                                         self.config.word_seq_length, self.config.phoneme_seq_length)
       
        print("final_feat:".format(final_feat))
        '''
        
        # add dropout
        with tf.name_scope("dropout"):
            final_feat_trs = tf.nn.dropout(final_feat_trs, self.keep_prob)
            final_feat_asr = tf.nn.dropout(final_feat_asr, self.keep_prob)

        with tf.name_scope("score"):
            # 分类器
            self.logits_trs = tf.layers.dense(final_feat_trs,
                                          self.num_classes,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.l2_reg_lambda),
                                          name='fc1')
            self.logits_asr = tf.layers.dense(final_feat_asr,
                                          self.num_classes,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.l2_reg_lambda),
                                          name='fc1',reuse=True)
            self.y_pred_cls_trs = tf.argmax(tf.nn.softmax(self.logits_trs), 1)  # 预测类别
            self.y_pred_cls_asr = tf.argmax(tf.nn.softmax(self.logits_asr), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            #one_hot_labels = tf.one_hot(self.input_y, depth=self.num_classes)
            cross_entropy_trs = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_trs, labels=self.input_y)
            cross_entropy_asr = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_asr, labels=self.input_y)
            self.loss_trs = tf.reduce_mean(cross_entropy_trs)
            self.loss_asr = tf.reduce_mean(cross_entropy_asr)
            
            if self.e3!=0:
                self.kl_loss = tf.reduce_mean(self.kl_loss_v3(self.logits_asr, self.logits_trs))
                #self.kl_loss = tf.reduce_mean(losses_kl)

                self.loss =  self.e1*self.loss_trs + self.e2*self.loss_asr + self.e3*self.kl_loss
            else:
                self.loss =  self.e1*self.loss_trs + self.e2*self.loss_asr
            
            #self.optim = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred_trs = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls_trs)
            self.accuracy_trs = tf.reduce_mean(tf.cast(correct_pred_trs, "float"), name="accuracy_trs")
            
            correct_pred_asr = tf.equal(tf.argmax(self.input_y, 1),self.y_pred_cls_asr)
            self.accuracy_asr = tf.reduce_mean(tf.cast(correct_pred_asr, "float"), name="accuracy_asr")
            

    def kl_loss_v3(self,y_pred,y_true):
        y_true = tf.nn.softmax(y_true)
        y_pred = tf.nn.softmax(y_pred)
        return tf.reduce_sum(y_true*tf.math.log(y_true/(y_pred+1e-10)+1e-10),axis=1)  #+1e-10
    
    
    def get_feed_dict(self, x_word, x_phoneme, x_word_len, x_phoneme_len, labels, keep_prob):
        feed_dict = {
            self.input_x_word: x_word,
            self.input_x_word_len: x_word_len,
            self.input_x_phoneme: x_phoneme,
            self.input_x_phoneme_len: x_phoneme_len,
            self.input_y: labels,
            self.keep_prob: keep_prob
        }
        return feed_dict

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(42)
import numpy as np
from numpy.random import seed
seed(42)

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length,
      num_classes,
      vocab_size,
      embedding_size, 
      filter_sizes, 
      num_filters,
      embeddings, 
      e1,
      e2,
      e3,
      l2_reg_lambda=0.0,
    ):


        nbest = 10
        self.input_x_asr = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_asr")
        
        self.input_x_trs = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_trs")
        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                #embeddings,  ## 使用embedding
                name="W")
            self.embedded_chars_asr = tf.nn.embedding_lookup(self.W, self.input_x_asr) ## 
#             self.embedded_chars_trs = tf.nn.embedding_lookup(self.W, self.input_x_trs)
            
            print(self.embedded_chars_asr.shape)  # [? , 800, 150]
            
            self.embedded_chars_asr = tf.transpose(self.embedded_chars_asr, [0,2,1])
            self.embedded_chars_asr_expanded = tf.expand_dims(self.embedded_chars_asr, -1)
            self.embedded_chars_asr_transpose = tf.transpose(self.embedded_chars_asr_expanded,[0,2,1,3])
            self.curr_split_data = tf.split(self.embedded_chars_asr_transpose, num_or_size_splits=10, axis=1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
      
        for i, filter_size in enumerate(filter_sizes):## 对每个filter，再一次for循环nbest
            
            curr_sentence_max = None
            ## for tensor1  in tensor 10: 10个句子 所以最终得到的shape是原来的10倍，应该在此方法前定义一个变量进行add 10个句子。
            for item in self.curr_split_data:
            
                with tf.name_scope("conv-maxpool-%s" % filter_size):

                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]  
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    ##-----------------------asr part------------------------------------------------- 
                    conv = tf.nn.conv2d(
                        item,  #self.embedded_chars_asr_expanded
                        W,
                        strides=[1, 1, 1, 1],  
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    '''
                    第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                    第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                    '''
#                     pooled_max = tf.nn.max_pool(  #  avg_pool  max_pool
#                         h,
#                         ksize=[1, sequence_length/10 - filter_size + 1, 1, 1],   
#                         strides=[1, 1, 1, 1],     
#                         padding='VALID',  # VALID  SAME
#                         name="pool")
                    
                    pooled_avg = tf.nn.avg_pool(  #  avg_pool  max_pool
                        h,
                        ksize=[1, sequence_length/10 - filter_size + 1, 1, 1],   
                        strides=[1, 1, 1, 1],     
                        padding='VALID',  # VALID  SAME
                        name="pool")

#                     if curr_sentence_max==None:
#                         curr_sentence_max=pooled_max
#                     else:
#                         curr_sentence_max = tf.maximum(curr_sentence_max,pooled_max)

                    if curr_sentence_max==None:
                        curr_sentence_max=pooled_avg
                    else:
                        curr_sentence_max = tf.maximum(curr_sentence_max,pooled_avg)
                    
            pooled_outputs.append(curr_sentence_max)
                                

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(self.h_pool_flat.shape)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores_asr = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores_asr")  
            
            self.predictions_asr = tf.argmax(self.scores_asr, 1, name="predictions_asr")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores_asr, labels=self.input_y)
            self.loss = tf.reduce_mean(loss) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions_asr = tf.equal(self.predictions_asr, tf.argmax(self.input_y, 1))
            self.accuracy_asr = tf.reduce_mean(tf.cast(correct_predictions_asr, "float"), name="accuracy_asr")

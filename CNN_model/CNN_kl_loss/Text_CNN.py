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
        
        # hyper para e1,e2,e3
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        
        embedding_size=embedding_size
        # Placeholders for input, output and dropout
        self.input_x_asr = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_asr")
        self.input_x_trs = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_trs")
        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                #embeddings,  ## 使用embedding
                name="W")
            self.embedded_chars_asr = tf.nn.embedding_lookup(self.W, self.input_x_asr)
            self.embedded_chars_trs = tf.nn.embedding_lookup(self.W, self.input_x_trs)

           
            self.embedded_chars_asr_expanded = tf.expand_dims(self.embedded_chars_asr, -1)
            self.embedded_chars_trs_expanded = tf.expand_dims(self.embedded_chars_trs, -1)
            

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        pooled_trs_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                
                ##-----------------------asr part------------------------------------------------- 
                conv = tf.nn.conv2d(
                    self.embedded_chars_asr_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                
                
                ##-------------------------trs part -----------------------------------------------
                conv = tf.nn.conv2d(
                    self.embedded_chars_trs_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled_trs = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_trs_outputs.append(pooled_trs)                
                ##------------------------------------------------------------------------
                
                
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        self.h_pool_trs = tf.concat(pooled_trs_outputs, 3)
        self.h_pool_trs_flat = tf.reshape(self.h_pool_trs, [-1, num_filters_total])        
        

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop_trs = tf.nn.dropout(self.h_pool_trs_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
#             l2_loss += tf.nn.l2_loss(W)
#             l2_loss += tf.nn.l2_loss(b)
            
            self.scores_asr = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores_asr")  
            self.scores_trs = tf.nn.xw_plus_b(self.h_drop_trs, W, b, name="scores_trs")       
            
            self.predictions_asr = tf.argmax(self.scores_asr, 1, name="predictions_asr")
            self.predictions_trs = tf.argmax(self.scores_trs, 1, name="predictions_trs")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            
            losses_asr = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores_asr, labels=self.input_y)
            
            losses_trs = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores_trs, labels=self.input_y)           
            
            #losses_kl = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores_asr, labels=self.scores_trs)      
            ## KL
            ## e1,e2,e3  trs,asr,KL
            self.loss_trs = tf.reduce_mean(losses_trs) 
            self.loss_asr = tf.reduce_mean(losses_asr)
            #self.kl_loss = tf.reduce_mean(self.kl_for_log_probs(self.scores_asr, self.scores_trs))#
            if self.e3!=0:
                self.kl_loss = tf.reduce_mean(self.kl_loss_v3(self.scores_asr, self.scores_trs))
                #self.kl_loss = tf.reduce_mean(losses_kl)

                self.loss =  self.e1*self.loss_trs + self.e2*self.loss_asr + self.e3*self.kl_loss
            else:
                self.loss =  self.e1*self.loss_trs + self.e2*self.loss_asr
            
            
            #KL_loss = kl(prob_a, prob_b)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions_asr = tf.equal(self.predictions_asr, tf.argmax(self.input_y, 1))
            self.accuracy_asr = tf.reduce_mean(tf.cast(correct_predictions_asr, "float"), name="accuracy_asr")
            
            correct_predictions_trs = tf.equal(self.predictions_trs, tf.argmax(self.input_y, 1))
            self.accuracy_trs = tf.reduce_mean(tf.cast(correct_predictions_trs, "float"), name="accuracy_trs")
            
            
            
    def kl(self,x, y):
        X = tf.distributions.Categorical(probs=tf.nn.softmax(x),allow_nan_stats=False)
        Y = tf.distributions.Categorical(probs=tf.nn.softmax(y),allow_nan_stats=False)
        return tf.distributions.kl_divergence(X, Y,allow_nan_stats=False)
    
    def kl_for_log_probs(self,log_p, log_q):
        p = tf.exp(log_p)
        neg_ent = tf.reduce_sum(p * log_p, axis=-1)
        neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
        kl = neg_ent - neg_cross_ent
        return kl
    
    def kl_loss_v3(self,y_pred,y_true):
        y_true = tf.nn.softmax(y_true)
        y_pred = tf.nn.softmax(y_pred)
        return tf.reduce_sum(y_true*tf.math.log(y_true/(y_pred+1e-10)+1e-10),axis=1)  #+1e-10
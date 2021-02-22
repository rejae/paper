# coding=utf-8
import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(42)
import pandas as pd
import numpy as np
from numpy.random import seed
seed(42)
import os
import sys
import codecs
import getopt
import time
import logging
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


log = logging.getLogger('CNNTrainAndTest')
log.setLevel(logging.DEBUG)
hdr = logging.StreamHandler(sys.__stdout__)
formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
hdr.setFormatter(formatter)
log.addHandler(hdr)



modelname = 'cnn'
rootdir = "data_newcnn"
timestamp = str(int(time.time()))
data_root_dir = "prepared"

labels_dir =  "data/cateAndQuest_waihu.txt"  # cateAndQuest  cateAndQuest_waihu

words_dir =  "data/words.txt"   #words.txt  vocabs
vectors_dir =  "data/vectors.txt"
model_file_dir = rootdir + "/model/"
if not os.path.exists(model_file_dir):
    os.makedirs(model_file_dir)
export_dir = model_file_dir + "/export/"

modelProp_dir = model_file_dir + "modelProp.txt"
cateProp_dir = model_file_dir + "cateProp.txt"
labelId_dir = model_file_dir + "label_id.txt"
wordId_dir = model_file_dir + "word_id.txt"
restlt_dir = model_file_dir + "result_"+modelname+".txt"

# Data loading params
tf.flags.DEFINE_string("modelname", "", "Upper param")
tf.flags.DEFINE_string("version", "", "Upper param")
tf.flags.DEFINE_string("rootdir", "", "Upper param")

tf.flags.DEFINE_string("m", "", "Upper param")
tf.flags.DEFINE_string("v", "", "Upper param")
tf.flags.DEFINE_string("r", "", "Upper param")

tf.flags.DEFINE_string("labels_dir", labels_dir, "label data")
tf.flags.DEFINE_string("words_dir", words_dir, "label data")
tf.flags.DEFINE_string("embedding_file", vectors_dir, "word2vec trained vector data")
tf.flags.DEFINE_string("export_path", export_dir, "word2vec trained vector data")
tf.flags.DEFINE_string("export_tag", "CNN", "train data")
tf.flags.DEFINE_integer("max_seq_len", 80, "max length of sequence")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")  #FLAGS.embedding_dim
tf.flags.DEFINE_string("filter_sizes", '2,3,4,5', "Comma-separated filter sizes (default: '3,4,5')") #"6,9,12,15"   
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_train_epochs", 100, "Number of train data set training epochs (default: 200)")   # 100
tf.flags.DEFINE_integer("num_all_epochs", 20, "Number of all data set training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("early_stop_epoch", 4, "early stop epoch")
tf.flags.DEFINE_float("max_accept_error", 0.001,"Maximum acceptable error")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("device_num", "", "How many Devices(cpu/gpu) to use.")
tf.flags.DEFINE_boolean("use_gpu", True, "whether to use gpu")


FLAGS = tf.flags.FLAGS
data_helper = data_helpers.JimiClassifyData(label_file = FLAGS.labels_dir, words_file = FLAGS.words_dir, max_sequence_len = FLAGS.max_seq_len)
# id2label = data_helper.id2labels
# print(id2label)
#embeddings = data_helpers.load_embeddings('w2v_all_embedding.txt') 

def preprocess(train,test):
    # Data Preparation
    # ==================================================

    # Load data
    log.info("Loading data...")
    train_x, train_x1,train_y, train_label_num,train_asr_data,train_trs_data = data_helper.load_data(train)
    test_x, test_x1,test_y, test_label_num,test_asr_data,test_trs_data  = data_helper.load_data(test)
    log.info("Train: %d, Test: %d" % (len(train_x), len(test_x)))
    print(train_x[1],train_y[1])
    print(test_x[1], test_y[1])
    return train_x,train_x1, train_y, test_x,test_x1, test_y,test_asr_data,test_trs_data

def generateReport(pre_cate, origin_cate_list):
#     print(pre_cate[0],origin_cate_list[0])
    origin_cate = []
    for tmp in origin_cate_list:
        label_index = int(np.argwhere(tmp == 1))
        origin_cate.append(label_index)
    origin_cate = np.asarray(origin_cate).astype(np.int32)
    
#     print(origin_cate[0])
#     print(np.argmax(pre_cate[0]))
#     print(origin_cate[0]==np.argmax(pre_cate[0]))
    count  = 0
    for i in range(len(origin_cate_list)):
        if origin_cate[i]==np.argmax(pre_cate[i]):
            count +=1
    return count/len(origin_cate_list)


def train(train_x,train_x1, train_y, test_x,test_x1, test_y,e1,e2,e3,filter_sizes,test_asr_data,test_trs_data):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=train_x.shape[1],
                num_classes=train_y.shape[1],
                vocab_size=len(data_helper.id2words),
                embedding_size=FLAGS.embedding_dim,
                
                filter_sizes = filter_sizes, # list(map(int, FLAGS.filter_sizes.split(","))),
                
                num_filters=FLAGS.num_filters,
                embeddings=None, #None,  #embeddings
                
                ## hyper para
                e1=e1,
                e2=e2,
                e3=e3,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
            
            )

            # Define Training procedure
            
            
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1*1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
#             loss_summary = tf.summary.scalar("loss", cnn.loss)
#             acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

#             # Train Summaries
#             train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
#             train_summary_dir = os.path.join(model_file_dir, "summaries", "train")
#             train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

#             # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             dev_summary_dir = os.path.join(model_file_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

#             # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
#             checkpoint_dir = os.path.abspath(os.path.join(model_file_dir, "checkpoints"))
#             checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            #np.savetxt("weight_init.txt", sess.run(cnn.W), delimiter=" ")
            def train_step(x_batch,x_batch_trs, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x_asr: x_batch,
                  cnn.input_x_trs: x_batch_trs,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
#                 _, step, loss, accuracy = sess.run(
#                     [train_op, global_step, cnn.loss, cnn.accuracy_asr],
#                     feed_dict)
                _, step, loss = sess.run(
                    [train_op, global_step, cnn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #log.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                log.info("{}: step {}, loss {:g}".format(time_str, step, loss))
                #train_summary_writer.add_summary(summaries, step)
            
            
            def testDataEvaluate(test_x, test_x1,test_y, writer=None):
                """
                Evaluates model on a test set
                """
                feed_dict = {
                  cnn.input_x_asr: test_x,
                  cnn.input_x_trs: None,
                  cnn.input_y: test_y,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss_asr, accuracy_asr, predictions_asr,scores_asr = sess.run(
                    [global_step, cnn.loss_asr, cnn.accuracy_asr, cnn.predictions_asr,cnn.scores_asr],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                log.info("----------------------------------------------------")
                log.info("test_asr {}: step {}, loss {:g}, acc {:g}, prediction legth {}".format(time_str, step, loss_asr,accuracy_asr, len(predictions_asr)))
                print(generateReport(scores_asr, test_y))
                with open('nbest-result_asr_oritest.txt','a',encoding='utf-8') as f:
                    f.write( str(e1) +"----"+ str(e2)+"----" + str(e3)+"----  " +str(generateReport(scores_asr, test_y))+'\n')
                
                ####################################################################################################
#                 if writer:
#                     writer.add_summary(summaries, step)

                feed_dict_trs = {
                  #cnn.input_x_asr: test_x,
                  cnn.input_x_trs: test_x1,
                  cnn.input_y: test_y,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss_trs, accuracy_trs, predictions_trs,scores_trs= sess.run(
                    [global_step, cnn.loss_trs, cnn.accuracy_trs, cnn.predictions_trs,cnn.scores_trs],feed_dict_trs)
                time_str = datetime.datetime.now().isoformat()
                log.info("----------------------------------------------------")
                log.info("test_trs {}: step {}, loss {:g}, acc {:g}, prediction legth {}".format(time_str, step, loss_trs, accuracy_trs, len(predictions_trs)))
                with open('nbest-result_trs_oritest.txt','a',encoding='utf-8') as f:
                    f.write(str(e1) +"----"+ str(e2)+"----" + str(e3)+"----  " +str(generateReport(scores_trs, test_y))+'\n')
                    
                print(generateReport(scores_trs, test_y))
                
                ######################################trs文本，trs预测，asr预测，##############################################################
                def get_label(pre_cate,origin_cate_list):
                    origin_cate = []
                    for tmp in origin_cate_list:
                        label_index = int(np.argwhere(tmp == 1))
                        origin_cate.append(label_index)
                    origin_cate = np.asarray(origin_cate).astype(np.int32)

                    is_equal=[]
                    predict_list = []
                    
                    for i in range(len(origin_cate_list)):
                        predict_list.append(np.argmax(pre_cate[i]))
                        
                        if origin_cate[i]!=np.argmax(pre_cate[i]):
                            is_equal.append(False)
                        else:
                            is_equal.append(True)
                    return is_equal,predict_list,origin_cate
                
#                 trs_is_equal,trs_predict_list,trs_origin_cate = get_label(scores_trs,test_y)
#                 asr_is_equal,asr_predict_list,asr_origin_cate = get_label(scores_asr,test_y)
                
#                 id2label = ['confirm', 'press', 'ask_identity', 'ask', 'deafness', 'busy', 'change_time', 'deny', 'return', 'ask_time', 'other',
#                             'ask_installation', 'change_address', 'confirm_buy', 'near_address', 'change']
#                 ##　test_asr_data,test_trs_data

#                 with open('result_analyze.txt','a',encoding='utf-8') as f:
#                     #f.write(str(e1) +"----"+ str(e2)+"----" + str(e3)+"----  " +str(generateReport(scores_trs, test_y))+'\n')
#                     f.write('test_trs_data trs_is_equal trs_predict_list trs_origin_cate test_asr_data asr_is_equal asr_predict_list asr_origin_cate '+'\n')
#                     for i in range(len(test_trs_data)):
#                         f.write(test_trs_data[i]+' '+str(trs_is_equal[i])+' '+id2label[trs_predict_list[i]]+' '+id2label[trs_origin_cate[i]]+' '+
#                                 test_asr_data[i]+' '+str(asr_is_equal[i])+' '+id2label[asr_predict_list[i]]+' '+id2label[asr_origin_cate[i]]+'\n')
                
            # trainDataSet training
            batches = data_helpers.batch_iter(
                list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_train_epochs)
            
            batches_trs = data_helpers.batch_iter(
                list(zip(train_x1, train_y)), FLAGS.batch_size, FLAGS.num_train_epochs)
            
            epoch_step = round(len(train_x)/FLAGS.batch_size)
            max_acc = 0.0
            num = 1
            break_epoch = 0
            for batch,batch_trs in zip(batches,batches_trs):
                x_batch, y_batch = zip(*batch)
                x_batch_trs,_ = zip(*batch_trs)
                
                train_step(x_batch,x_batch_trs, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                
#                 if current_step % FLAGS.checkpoint_every == 0:
#                     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
#                     log.info("Saved model checkpoint to {}\n".format(path))
                '''if current_step % epoch_step == 0:
                    log.info("Begin to evalute test_dataset.")
                    test_df = testDataEvaluate(x_test, y_test, writer=dev_summary_writer)
                    tmp_acc = (test_df['predictions'] == test_df['labels']).mean()
                    log.info('test accuracy: %f, epoch: %d' % (tmp_acc, current_step / epoch_step))
                    if tmp_acc - max_acc > FLAGS.max_accept_error:
                        num = TextCNN
                        max_acc = tmp_acc
                    else:
                        num = num + 1
                if num > FLAGS.early_stop_epoch :
                    log.info('trigger early stop')
                    break_epoch = current_step/epoch_step
                    log.info('break_epoch: %d, max_acc: %f' % (break_epoch, max_acc))
                    break'''
                
            testDataEvaluate(test_x,test_x1, test_y, writer=None)#dev_summary_writer


    log.info("CNN train finish!")
            
            
def main(argv=None):
      #  trs  asr kl
        
    a,b,c = [1,0,1,1],[0,1,1,0],[0,0,0,1]

    a,b,c = [1],[0],[0]
#     a,b,c = [0],[1],[0]
#     a,b,c = [1],[1],[0]
#     a,b,c = [1],[0],[1]
    for i,j,k in zip(a,b,c):
        e1,e2,e3=i,j,k
        train_names = ['train_asr_trs_trsl_clean_2.csv'] 
        test_names = ['test_asr_trs_trsl_clean.csv']  
#         train_names = ['nbest_train_data.csv','nbest_train_data.csv','nbest_train_data.csv','nbest_train_data.csv','nbest_train_data.csv'] 
#         test_names = ['test_asr_trs_trsl_clean.csv','test_asr_trs_trsl_clean.csv','test_asr_trs_trsl_clean.csv','test_asr_trs_trsl_clean.csv','test_asr_trs_trsl_clean.csv']  
        count = 0
        for train_name,test_name in zip(train_names,test_names):
            count+=1
            train_x,train_x1, train_y, test_x,test_x1, test_y,test_asr_data,test_trs_data= preprocess('data/'+train_name,'data/'+test_name)

            train(train_x, train_x1, train_y, test_x, test_x1, test_y,e1,e2,e3, [2,3,4,5],test_asr_data,test_trs_data)

if __name__ == '__main__':
    tf.app.run()

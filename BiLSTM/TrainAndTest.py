# coding=utf-8
import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(666)
import pandas as pd
import numpy as np
from numpy.random import seed
seed(666)
import os
import sys
import codecs
import getopt
import time
import logging
import datetime
import data_helpers
from text_cnn import TextCNN
from bi_lstm_model import TextBiRNN
from tensorflow.contrib import learn


log = logging.getLogger('TrainAndTest')
log.setLevel(logging.DEBUG)
hdr = logging.StreamHandler(sys.__stdout__)
formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
hdr.setFormatter(formatter)
log.addHandler(hdr)


dataType='waihu'

modelname = 'BiLstm'
rootdir = "output"
timestamp = str(int(time.time()))
labels_dir =  "prepared/cateAndQuest_"+dataType+".txt"  # cateAndQuest  cateAndQuest_waihu
words_dir =  "prepared/word_"+dataType+".txt"   #words.txt  vocabs

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
tf.flags.DEFINE_string("export_path", export_dir, "word2vec trained vector data")
tf.flags.DEFINE_string("export_tag", "BiLstm", "train data")
tf.flags.DEFINE_integer("word_seq_length", 40, "max length of sequence")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")  #FLAGS.embedding_dim
tf.flags.DEFINE_string("filter_sizes", '2,3,4,5', "Comma-separated filter sizes (default: '3,4,5')") #"6,9,12,15"   
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_train_epochs", 20, "Number of train data set training epochs (default: 200)")   # 100
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
data_helper = data_helpers.JimiClassifyData(label_file = FLAGS.labels_dir, words_file = FLAGS.words_dir, max_sequence_len = FLAGS.word_seq_length)
# id2label = data_helper.id2labels
# print(id2label)
#embeddings = data_helpers.load_embeddings('w2v_all_embedding.txt') 

def preprocess(train,dev,test):
    # Data Preparation
    # ==================================================

    # Load data
    log.info("Loading data...")
    train_x_trs, train_x_asr,train_y = data_helper.load_data(train,dataType)
    dev_x_trs, dev_x_asr,dev_y = data_helper.load_data(dev,dataType)
    test_x_trs, test_x_asr,test_y = data_helper.load_data(test,dataType)
    log.info("Train: %d,  Dev: %d, Test: %d" % (len(train_x_trs),len(dev_x_trs), len(test_x_trs)))
    
    
    return train_x_trs, train_x_asr,train_y, dev_x_trs, dev_x_asr,dev_y,test_x_trs, test_x_asr,test_y

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


def train(train_x_trs, train_x_asr,train_y, dev_x_trs, dev_x_asr,dev_y,test_x_trs, test_x_asr,test_y,e1,e2,e3,count):
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            #--------------------------------------CNN---------------------------------------
            '''
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
            '''
            #--------------------------------------BiLstm---------------------------------------
            model=TextBiRNN(
                FLAGS,
                sequence_length=FLAGS.word_seq_length,num_classes=train_y.shape[1],vocab_size=len(data_helper.id2words),
                embedding_size=FLAGS.embedding_dim,embeddings=None, hidden_dim=FLAGS.embedding_dim,
                ## hyper para
                e1=e1,
                e2=e2,
                e3=e3,
                
            )
            
            
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1*1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            #train_op = optimizer.minimize(model.loss)

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
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy_asr)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(model_file_dir,str(count), "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(model_file_dir,str(count), "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(model_file_dir,str(count), "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            #np.savetxt("weight_init.txt", sess.run(cnn.W), delimiter=" ")
            def train_step(x_batch_trs,x_batch_asr, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  model.input_x_trs: x_batch_trs,
                  model.input_x_asr: x_batch_asr,
                  model.input_y: y_batch,
                  model.keep_prob: FLAGS.dropout_keep_prob
                }
#                 _, step, loss, accuracy = sess.run(
#                     [train_op, global_step, cnn.loss, cnn.accuracy_asr],
#                     feed_dict)
                _, step, loss = sess.run(
                    [train_op, global_step, model.loss],
                    feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                #log.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                log.info("{}: step {}, loss {:g}".format(time_str, step, loss))
                #train_summary_writer.add_summary(summaries, step)
            
            
            def testDataEvaluate(test_x_trs, test_x_asr,test_y, writer=None):
                """
                Evaluates model on a test set
                """
                feed_dict = {
                  model.input_x_asr: test_x_asr,
                  #model.input_x_trs: test_x1,
                  model.input_y: test_y,
                  model.keep_prob: 1.0
                }
                step, loss_asr, accuracy_asr, predictions_asr,scores_asr = sess.run(
                    [global_step, model.loss_asr, model.accuracy_asr, model.y_pred_cls_asr,model.logits_asr],
                    feed_dict)

                ####################################################################################################
#                 if writer:
#                     writer.add_summary(summaries, step)
    
                #-------------------------------------------1 0 1---------------------------
                '''
                feed_dict_trs = {
                  #model.input_x_asr: test_x_trs,
                  model.input_x_trs: test_x_trs,
                  model.input_y: test_y,
                  model.keep_prob: 1.0
                }
                step, loss_trs, accuracy_trs, predictions_trs,scores_trs= sess.run(
                    [global_step, model.loss_trs, model.accuracy_trs, model.y_pred_cls_trs,model.logits_trs],feed_dict_trs) 
                
                return generateReport(scores_trs, test_y),generateReport(scores_asr, test_y)
                '''
                #-------------------------------------------0 1 0---------------------------
               
                feed_dict_trs = {
                  model.input_x_asr: test_x_trs,
                  #model.input_x_trs: test_x_trs,
                  model.input_y: test_y,
                  model.keep_prob: 1.0
                }
                step, loss_trs, accuracy_trs, predictions_trs,scores_trs = sess.run(
                    [global_step, model.loss_asr, model.accuracy_asr, model.y_pred_cls_asr,model.logits_asr],
                    feed_dict_trs)
                
                return generateReport(scores_trs, test_y),generateReport(scores_asr, test_y)
            
            # trainDataSet training
            batches_asr = data_helpers.batch_iter(
                list(zip(train_x_asr, train_y)), FLAGS.batch_size, FLAGS.num_train_epochs)
            
            batches_trs = data_helpers.batch_iter(
                list(zip(train_x_trs, train_y)), FLAGS.batch_size, FLAGS.num_train_epochs)
            
            epoch_step = round(len(train_x_trs)/FLAGS.batch_size)
            max_acc_asr = 0.0
            max_acc_trs = 0.0
            num = 0
            max_epoch =0
            break_epoch = 0
            for batch_asr,batch_trs in zip(batches_asr,batches_trs):
                x_batch_asr, y_batch = zip(*batch_asr)
                x_batch_trs,_ = zip(*batch_trs)
                
                train_step(x_batch_trs,x_batch_asr,y_batch)
                current_step = tf.train.global_step(sess, global_step)
                
                if current_step % epoch_step == 0:
                    log.info("Begin to evalute dev_dataset.")
                    num+=1
                    dev_df_trs,dev_df_asr = testDataEvaluate(dev_x_trs, dev_x_asr,dev_y,writer=None)
                    log.info('dev trans accuracy: %f, dev asr accuracy: %f, epoch: %d' % (dev_df_trs,dev_df_asr, current_step / epoch_step))
                        
                    if dev_df_asr > max_acc_asr:
                        max_epoch=num
                        max_acc_asr=dev_df_asr
                        max_acc_trs=dev_df_trs
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        log.info("Saved model checkpoint to {}\n".format(path))
            
            print('----------------dev best asr:'+str(max_acc_asr)+',trans:'+str(max_acc_trs)+',best epoch:'+str(max_epoch)+'------------------------'+'\n')
            ckpt_file = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess,ckpt_file)
            print(testDataEvaluate(dev_x_trs, dev_x_asr,dev_y,writer=None))
            with open(dataType+'_result.txt','a',encoding='utf-8') as f:
                f.write('----------------dev best asr:'+str(max_acc_asr)+',trans:'+str(max_acc_trs)+',best epoch:'+str(max_epoch)+'---------------'+'\n')
                f.write('---------------test----------------------'+'\n')
                test_df_trs,test_df_asr = testDataEvaluate(test_x_trs, test_x_asr,test_y,writer=None)
                f.write("trans:"+str(test_df_trs)+'\n')
                f.write("asr:"+str(test_df_asr)+'\n')

    log.info("Finish!")
            
            
def main(argv=None):
    #a,b,c =[1],[0],[1]
    a,b,c =[0],[1],[0]
    for i,j,k in zip(a,b,c):
        repeat=3
        e1,e2,e3=i,j,k
        train_names = [dataType+'_phoneme_datas_train.txt']*repeat 
        test_names = [dataType+'_phoneme_datas_test.txt']*repeat  
        dev_names = [dataType+'_phoneme_datas_dev.txt']*repeat 
        count = 0
        for train_name,dev_name,test_name in zip(train_names,dev_names,test_names):
            count+=1
            train_x_trs,train_x_asr, train_y,dev_x_trs,dev_x_asr, dev_y,test_x_trs,test_x_asr, test_y= preprocess('data/'+train_name,'data/'+dev_name,'data/'+test_name)

            train(train_x_trs,train_x_asr, train_y,dev_x_trs,dev_x_asr, dev_y,test_x_trs,test_x_asr, test_y,e1,e2,e3,count)

if __name__ == '__main__':
    tf.app.run()

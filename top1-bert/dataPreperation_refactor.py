# coding=utf-8
import os
import sys
import random
import codecs
import getopt



outdir='/home/wangzexun/notespace/qinjie/Top1_Classify/result'


# train_names=[ "train_"+i for i in ['1_88k_hyp.txt','2_64k_hyp.txt','3_64k_trs.txt','4_88_64_hpy.txt']] 
# test_names =[ "test_24k.txt" for _ in range(4)]  #_24k


### 交叉拼接实验
train_names=["cross_trs_phoneme_train.txt","cross_asr_phoneme_train.txt","cross_trs_train.txt","cross_asr_train.txt"]
test_names = ["cross_trs_phoneme_test.txt","cross_trs_phoneme_test.txt",'cross_trs_test.txt','cross_trs_test.txt']

### 直接拼接实验
# train_names=["trs_phoneme_train.txt","asr_phoneme_train.txt","trs_train.txt","asr_train.txt"]
# test_names = ["trs_phoneme_test.txt","trs_phoneme_test.txt","trs_test.txt","trs_test.txt"]


for train_name,test_name in zip(train_names,test_names):
    train_set = []
    test_set = []
    print(train_name,test_name)
    datadir='/home/wangzexun/notespace/qinjie/Top1_Classify/data'
    with codecs.open(os.path.join(datadir, train_name), 'r', 'utf-8') as inf:
        for line in inf:
            ws = line.strip().split('\t')

            if len(ws) == 3 and ws[1] != '':
                train_set.append("{}\t{}".format(ws[2], ws[1]))   
    with codecs.open(os.path.join(datadir, test_name), 'r', 'utf-8') as inf:
        for line in inf:
            ws = line.strip().split('\t')
            if len(ws) == 3 and ws[1] != '':
                test_set.append("{}\t{}".format(ws[2], ws[1]))   

    random.seed(42)
    random.shuffle(train_set)
    #random.shuffle(test_set)    

    with codecs.open(os.path.join(outdir, 'prepared', 'train', 'corpus.tsv'), 'w', 'utf-8') as outf:
      for line in train_set:
        outf.write("T\t{}\n".format(line))   
      for line in test_set:
        outf.write("D\t{}\n".format(line))   
        outf.write("E\t{}\n".format(line))      

    with codecs.open(os.path.join(outdir, 'prepared', 'train', 'cateAndQuest.txt'), 'w', 'utf-8') as outf:
        with codecs.open(os.path.join(datadir,'cateName.txt'), 'r', 'utf-8') as inf:
            for line in inf:
              outf.write(line)
    
    ## 合并 train_and_test文件
    import os
    import sys
    import getopt
    import subprocess

    # Train
    for i in range(3):
        datadir = os.path.join(outdir, "prepared", "train")
        modeldir = os.path.join(outdir,  train_name[:-4]+'_curr_'+str(i)) #模型名字为train_set+'i'
        script = "python3 cpu_classifier.py --do_train=true --do_predict=true --do_eval=true " \
                 " --data_dir={}/ --vocab_file=data/vocab.txt " \
                 "--bert_config_file=data/bert_config.json " \
                 "--init_checkpoint=waihu_pretrain_model_new/model.ckpt-188000 " \
                 "--max_seq_length=50 --batch_size=64 " \
                 "--predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 " \
                 "--output_dir={}/".format(datadir, modeldir)

        ret = subprocess.call(script, shell=True)
        if ret != 0:
            raise Exception("Command failed with return code {}".format(ret))
        print("训练完毕"+str(i))
    
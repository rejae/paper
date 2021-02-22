# coding=utf-8
import os
import sys
import random
import codecs
import getopt



outdir='/home/wangzexun/notespace/qinjie/Concatenate_Classify_MoreFeature_weighting/result'


train_names=[ "v3_train_set"+str(i)+".txt" for i in range(2,12,2)]
test_names =[ "v3_test_set"+str(i)+".txt" for i in range(2,12,2)]
# train_names=["train.txt"]
# test_names = ["test.txt"]
for train_name,test_name in zip(train_names,test_names):
    train_set = []
    test_set = []
    print(train_name,test_name)
    datadir='/home/wangzexun/notespace/qinjie/Concatenate_Classify_MoreFeature_weighting/data'
    # 85079743_1	[unused1] 嗯	0	1.0 1.0	confirm
    
    #88372770	[unused1] 对	confirm	0	1.0 1.0

    with codecs.open(os.path.join(datadir, train_name), 'r', 'utf-8') as inf:
        for line in inf:
            ws = line.strip().split('\t')
            #print('xxxxxxxxxxxxxxx',len(ws))
            if len(ws) == 5 and ws[1] != '':
                train_set.append("{}\t{}\t{}\t{}".format(ws[2], ws[1], ws[3], ws[4]))   
                #print(train_set)
    with codecs.open(os.path.join(datadir, test_name), 'r', 'utf-8') as inf:
        for line in inf:
            ws = line.strip().split('\t')
            if len(ws) == 5 and ws[1] != '':
                test_set.append("{}\t{}\t{}\t{}".format(ws[2], ws[1], ws[3], ws[4]))    

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
        modeldir = os.path.join(outdir,  ''.join(train_name[:-4])+'xxxxxxrepeat_lm'+str(i)) #模型名字为train_set+'i'
#         script = "python3 cpu_classifier.py --do_train=true --do_predict=true --do_eval=false " \
#                  " --data_dir={}/ --vocab_file=data/vocab.txt " \
#                  "--bert_config_file=data/bert_config.json " \
#                  "--init_checkpoint=waihu_pretrain_model_new/model.ckpt-188000 " \
#                  "--max_seq_length=120 --batch_size=64 " \
#                  "--predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 " \
#                  "--output_dir={}/".format(datadir, modeldir)
        
        script = "python3 cpu_classifier.py --do_train=true --do_predict=true --do_eval=false " \
         " --data_dir={}/ --vocab_file=data/vocab.txt " \
         "--bert_config_file=data/bert_config.json " \
         "--init_checkpoint=waihu_pretrain_model_new/model.ckpt-188000 " \
         "--max_seq_length=150 --batch_size=64 " \
         "--predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 " \
         "--output_dir={}/".format(datadir, modeldir)

        ret = subprocess.call(script, shell=True)
        if ret != 0:
            raise Exception("Command failed with return code {}".format(ret))
        print("训练完毕"+str(i))
    
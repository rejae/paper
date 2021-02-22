  # coding=utf-8
import os
import sys
import getopt
import subprocess

outdir='/home/wangzexun/notespace/qinjie/Concatenate_Classify/result'

# Train
datadir = os.path.join(outdir, "prepared", "train")
modeldir = os.path.join(outdir, "middle6604", "train")
script = "python3 cpu_classifier.py --do_train=true --do_predict=true --do_eval=false " \
         " --data_dir={}/ --vocab_file=data/vocab.txt " \
         "--bert_config_file=data/bert_config.json " \
         "--init_checkpoint=waihu_pretrain_model_new/model.ckpt-188000 " \
         "--max_seq_length=120 --batch_size=64 " \
         "--predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 " \
         "--output_dir={}/".format(datadir, modeldir)
ret = subprocess.call(script, shell=True)
if ret != 0:
	raise Exception("Command failed with return code {}".format(ret))
print("训练完毕")

# coding=utf-8
import os
import sys
import random
import codecs
import getopt


datadir='/home/wangzexun/notespace/wangzexun/project/N-Best/Concatenate_Classify_MoreFeature_weighting/data'
outdir='/home/wangzexun/notespace/wangzexun/project/N-Best/Concatenate_Classify_MoreFeature_weighting/result'

train_set = []
test_set = []
with codecs.open(os.path.join(datadir, 'train.txt'), 'r', 'utf-8') as inf:
    for line in inf:
        ws = line.strip().split('\t')
        if len(ws) == 5 and ws[1] != '':
            train_set.append("{}\t{}\t{}\t{}".format(ws[4], ws[1], ws[2], ws[3]))   
with codecs.open(os.path.join(datadir, 'test.txt'), 'r', 'utf-8') as inf:
    for line in inf:
        ws = line.strip().split('\t')
        if len(ws) == 5 and ws[1] != '':
            test_set.append("{}\t{}\t{}\t{}".format(ws[4], ws[1], ws[2], ws[3]))      
print(len(train_set))
print(len(test_set))

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
          
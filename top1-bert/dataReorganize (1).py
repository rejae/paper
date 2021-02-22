# coding=utf-8
import os
import sys
import codecs
import getopt
import numpy as np
from shutil import copyfile
from distutils.dir_util import copy_tree

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "m:v:r:", ["modelname=", "version=", "rootdir="])
except getopt.GetoptError:
    print('parameter error')
    sys.exit(2)
for opt, arg in opts:
    if opt in ['-m', '--modelname']:
        modelname = arg
    elif opt in ['-v', '--version']:
        version = arg
    elif opt in ['-r', '--rootdir']:
        rootdir = arg
print('model:%s' % modelname)
print('version:%s' % version)
print('rootdir:%s' % rootdir)

#os.mkdir(os.path.join(rootdir, 'model'))

# Generate label
id2label = dict()
with codecs.open(os.path.join(rootdir, 'data', 'cateName.txt'), 'r', 'utf-8') as inf, \
        codecs.open(os.path.join(rootdir, 'model', 'label_id.txt'), 'w', 'utf-8') as outf:
    for line in inf:
        ws = line.strip().split('\t')
        if len(ws) > 1:
            outf.write("{}\t{}\t{}\n".format(ws[0], len(id2label), ws[1]))
        else:
            outf.write("{}\t{}\t{}\n".format(ws[0], len(id2label), ws[0]))
        id2label[len(id2label)] = ws[0]

# Generate prop
test_score_file = os.path.join(rootdir, 'middle', 'train', 'test_results.tsv')
label_count = dict()
with codecs.open(test_score_file, 'r', 'utf-8') as inf:
    for line in inf:
        ws = line.strip().split('\t')

        label_exp = ws[0]
        label_act = id2label[np.argmax(np.array(ws[1:]).astype(np.float))]

        if not label_exp in label_count:
            label_count[label_exp] = []
            print("Expect {} Get {}".format(label_exp,label_act))
        label_count[label_exp].append(label_exp == label_act)

items_total = []
items_busi = []
items_other = []
for label in label_count.keys():
    items_total.extend(label_count[label])
    if label == 'other':
        items_other.extend(label_count[label])
    else:
        items_busi.extend(label_count[label])
items_total = np.array(items_total)
items_busi = np.array(items_busi)
items_other = np.array(items_other)

with codecs.open(os.path.join(rootdir, 'model', 'modelProp.txt'), 'w', 'utf-8') as outf:
    outf.write("{}\t{}\t{}\t0.6".format(np.average(items_total), np.average(items_busi), np.average(items_other)))
    
with codecs.open(os.path.join(rootdir, 'model', 'cateProp.txt'), 'w', 'utf-8') as outf:
    for label in label_count.keys():
        outf.write("{}\t{}\t{}\n".format(modelname, label, np.average(label_count[label])))
        
# Copy model
model_root = os.path.join(rootdir, 'middle', 'all', 'model')
copyfile(os.path.join(model_root, 'config.json'), os.path.join(rootdir, 'model', 'config.json'))
copyfile(os.path.join(model_root, 'vocab'), os.path.join(rootdir, 'model', 'words_id.txt'))

"""
model_root = [f.path for f in os.scandir(model_root) if f.is_dir()][0]
copyfile(os.path.join(model_root, 'saved_model.pb'), os.path.join(rootdir, 'model', 'saved_model.pb'))
copy_tree(os.path.join(model_root, 'variables'), os.path.join(rootdir, 'model', 'variables'))
"""

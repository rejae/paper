# coding=utf-8
import numpy as np
import codecs
import logging
import sys

#np.random.seed(42)
log = logging.getLogger('CNNTrainAndTest.data_helpers')

def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_embeddings(embedding_file):
    word2vec_embeddings = np.array([ [float(v) for v in line.strip().split(' ')] for line in codecs.open(embedding_file, 'r', 'utf-8').readlines()], dtype=np.float32)
    embedding_size = word2vec_embeddings.shape[1]
    unknown_padding_embedding = np.random.normal(0, 0.1, (2,embedding_size))

    embeddings = np.append(word2vec_embeddings, unknown_padding_embedding.astype(np.float32), axis=0)

    return embeddings

class JimiClassifyData(object):
    def __init__(self,label_file=None,
                 words_file=None,
                 max_sequence_len=80):
        self.max_sequence_len = max_sequence_len

            
        self.id2labels = [line.split("\t")[0].strip() for line in codecs.open(label_file, 'r', 'utf-8').readlines()]
        #self.labe2questions = [line2.split("\t")[1].strip() for line2 in codecs.open(label_file, 'r', 'utf-8').readlines()]
        self.label2ids = {x : i for i, x in enumerate(self.id2labels)}
        
        self.id2words = [line.split(' ')[0] for line in codecs.open(words_file, 'r', 'utf-8').readlines()]
        #self.id2words.append('UNKNOWN')
        #self.id2words.append('<PAD>')
        self.word2ids = {x : i for i, x in enumerate(self.id2words)}
        self.unknown_id = self.word2ids['UNKNOWN']
        self.padding_id = self.word2ids['<PAD>']
    
#     def load_data(self, inFilePath, otherNum):
#         x_data = []
#         y_data = []
#         label_num = {}

#         fIn = codecs.open(inFilePath, 'r', 'utf-8')
#         other_num = 0
#         for line in fIn:
#             line = line.strip()
#             fields = line.split(" ")
#             if len(fields) < 2:
#                 continue
#             label = fields[-1].strip()
#             y_data.append(np.arange(len(self.id2labels)) == self.label2ids[label])
            
#             if label == 'other':
#                 if other_num >= otherNum:
#                     continue
#                 else:
#                     other_num += 1

#             if label not in self.id2labels:
#                 log.info('invalid label:%s' % line)
#                 log.info('label:%s' % label)
#                 continue
#             content = line[:len(line)-len(label)].strip()
#             tokenIds, tokenStr = self.format_sentence(content)
#             x_data.append(tokenIds)

#             if label in label_num:
#                 label_num[label] += 1
#             else:
#                 label_num[label] = 1
	    
#         fIn.close()
#         x = np.array(x_data)
#         y = np.array(y_data).astype(np.int32)
#         #ins_num = len(y_data)
#         #label_weight = [0]*len(self.label2ids)
#         #label_weight = np.array([ pow((1-label_num[label]*1.0/ins_num), 3) for label in self.id2labels ], dtype=np.float32)

#         return x, y, label_num

    def load_data(self,inFilePath):
        x_asr_data = []
        x_trs_data = []
        x_asr_data_real=[]
        x_trs_data_real=[]
        y_data = []
        label_num = {}

        fIn = codecs.open(inFilePath, 'r', 'utf-8')
        for line in fIn:
            line = line.strip()
            fields = line.split(",")

            label = fields[-1].strip()
#             asr_data = fields[0]
#             trs_data = fields[1]
            asr_data = fields[1]
            trs_data = fields[2]
#             if label=='other':
#                 print(line)
            y_data.append(np.arange(len(self.id2labels)) == self.label2ids[label])

            tokenIds_asr, tokenStr_asr = self.format_sentence(asr_data)
            tokenIds_trs, tokenStr_trs = self.format_sentence(trs_data)

            x_asr_data.append(tokenIds_asr)
            x_trs_data.append(tokenIds_trs)
            
            x_asr_data_real.append(asr_data.replace(' ',''))
            x_trs_data_real.append(trs_data.replace(' ',''))


        fIn.close()
        x1 = np.array(x_asr_data)
        x2 = np.array(x_trs_data)

        y = np.array(y_data).astype(np.int32)

        return x1,x2, y, label_num ,x_asr_data_real,x_trs_data_real
    def get_label_weight(self, label_num, ins_num):
        label_weight = [0]*len(self.label2ids)
        label_weight = np.array([ pow((1-label_num[label]*1.0/ins_num), 1) for label in self.id2labels ], dtype=np.float32)
        return label_weight

    
    def format_sentence(self, tokens):
        token_fie = []
        if tokens != '':
            token_fie = tokens.split(' ')
        token_res = token_fie[:self.max_sequence_len]

        if len(token_res) < self.max_sequence_len:
            token_res += ['<PAD>']*(self.max_sequence_len - len(token_res))
        tokenStr = ' '.join(token_res)

        tokenIds = []
        for token in token_res:
            if token in self.word2ids:
                tokenIds.append(self.word2ids[token])
            else:
                tokenIds.append(self.unknown_id)
        return tokenIds, tokenStr

    


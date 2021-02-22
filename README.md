## 论文归档

# ASR文本鲁棒性SLU研究

## 1. KL loss 对抗训练
《Towards an ASR error robust Spoken Language Understanding System》
通过计算trans label 和 ASR pred_label 的KL loss来拉齐优化pred的参数

## 2. 在预训练段对相似易混token进行cosine_similarity计算loss加入到语言模型的loss中优化embedding表征
《LEARNING ASR-ROBUST CONTEXTUALIZED EMBEDDINGS FOR SPOKEN LANGUAGE UNDERSTANDING》

使用elmo模型，采用ca_finetune即confusion aware的finetune, 拉齐相似字的embedding表征。实验结果有一定的提升，但是与nbest模型比较还是显劣势。


## 3. lattice RNN 使用lattice数据进行预训练和预测 
《LATTICERNN: Recurrent Neural Networks over Lattices》  --lattice结构引入

《Self-Attentional Models for Lattice Inputs》

《ADAPTING PRETRAINED TRANSFORMER TO LATTICES FOR SPOKEN LANGUAGE UNDERSTANDING》

《Learning Spoken Language Representations with Neural Lattice Language Modeling》

预测精度提高 速度显著下降  主要是 lattice的batch数据中，每个时间步的短板决定了该时间步所需要的时间

## 4. 使用N-best数据
《IMPROVING SPOKEN LANGUAGE UNDERSTANDING BY EXPLOITING ASR N-BEST HYPOTHESES》

Nbest数据在embedding层面（更好）或者文本层面的拼接，显然的对预测结果有很大的提升，但是N-best数据中应该有一个权重调整，可结合am lm得分进行加权

## 5. 多任务模型
《Multi-task Learning of Spoken Language Understanding by Integrating N-Best Hypotheses with Hierarchical Attention》

This work is motivated by introducing multi-task learning (MTL), transfer learning (TL) and acoustic-model information into the framework of integrating n-best hypotheses for spoken language understanding. Among those algorithms, we find the MTL results in higher performance compared to the TL

- The left side is training all tasks (TR, DC, IC) in the same stage.
- right side is to train TR firstly and fine-tune or generate texts base on the pre-trained TR model for DC and IC.

Transcription Reconstruction (TR)

Domain Classification (DC)

Intent Classification (IC)

《Data balancing for boosting performance of low-frequency classes in Spoken Language Understanding》
针对数据不平衡的问题，采用多任务框架，主任务和辅助任务使用同一个SLU系统，但是辅助任务对数据中标签的分布做了调整，使用了一个特别的batch_data_balanced_generator

该论文讨论了针对数据不平衡采用的一些方法，包括：
- 过采样
- 带权loss
- 数据增强
- 类别平衡的batch generator
由于常用的数据平衡方法如过采样，带权loss可能导致过拟合问题，论文提出了使用多任务机制来解决过拟合问题。主任务和辅助任务共享权重参数，但在推断的时候仅使用主任务。


## 6. 应用新的模型GCN

《Short Text Classification using Graph Convolutional Network》

GCN模型节点为句子的token 和 label, token与token 节点之间的边通过词共现计算，token与label之间的节点关系通过token的频率和label的频率计算

- 短文本，缺乏语法结构
- 短文本，表达方式多样
- PTC中，命名差异大但是是同种类别

相似发音的易混字成了不同的节点，节点数目会显著增长，如何解决相似发音节点的融合是一个复杂的问题。



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Change the path in the gramsFilePath to any path to store bigrams file

"""

rnn_layers = 3
rnn_size = 512
EMBED_DIMENSION = 100
empty=0
eos=1
unk=2
seed=70
batch_norm=False
nflips=10
# training parameters

p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=1
#nflips=10 #changed from 10
#seed=72
regularizer = None
numOfGrams = 2
SLIDING_WINDOW_SIZE = 3

FN0 = 'pre_process'  # filename of vocab embeddings
FN1 = 'training'  # filename of model weights




#inputFile = "/Users/prathibha/Documents/Project/Try.json" 
#basePath = "/Users/prathibha/Documents/Project"
#dataPath = basePath + '/data'
#embeddingsMetadataFilePath = dataPath + 'metadata.tsv'
#embeddingsFilePath = dataPath + 'embeddings.tsv'
gramsFilePath ="/Users/prathibha/Documents/Project/bigrams/gramsDict.txt"



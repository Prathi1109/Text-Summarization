#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
import h5py
import keras.backend as K
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from generate_samples import gensamples
from generate_summary import gen
from pre_process import createEmbeddingModel
from constants import seed,batch_size,LR,rnn_size,rnn_layers,regularizer,optimizer,EMBED_DIMENSION,p_emb,p_W, p_U,p_dense,FN0,FN1
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.models import load_model
#from keras.optimizers import adam

from NN_model import SimpleContext,simple_context,create_model,inspect_model
#import sys
#sys.stdout = open('/Users/prathibha/Documents/Project/projectTest1.txt', 'w')
def load_weights(model, filepath):
    """Load all weights possible into model from filepath.
    This is a modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """

    #print('Loading', filepath, 'to', model.name)  
   
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        print("Layers:")
        print(layer_names)
        print("#######")
        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                
                try:
                    layer = model.get_layer(name=name)
                    print("layers exist",layer)
                except:
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
    #print("weight_values",weight_values)            
    return weight_values
if __name__== "__main__":
    
    CONTENT_SEQ_LEN,DESC_SEQ_LEN,maxlen,embedModel,contentlist,desclist,w2v,word2index,index2word= createEmbeddingModel( "/content/gdrive/My Drive/CS_training/data/DataSet/CStrain/",0,0)

    
    ##print("CONTENT_SEQ_LEN",CONTENT_SEQ_LEN)
    #print("DESC_SEQ_LEN",DESC_SEQ_LEN)
    VOCAB_SIZE=len(embedModel.wv.vocab)+5
    #print("VOCAB_LEN",VOCAB_SIZE)
    activation_rnn_size = 25 if CONTENT_SEQ_LEN else 0
    
    X=[]
    for item in contentlist:  
        item_list=[]
        for token in item:
            try:
                item_list.append(word2index[token])
            except:
                item_list.append(2)
                ##print("token not found"+token)
        X.append(item_list)   
    #print(X)
    
        
    Y=[]
    for item in desclist:  
        item_list=[]
        for token in item:
            try:
                item_list.append(word2index[token])
            except:
                item_list.append(2)
               # #print("token not found"+token)
        Y.append(item_list) 
    #print(Y)
    

    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=seed)  #Prathibha 20% of data to test
    
    #print("X_train Length " ,len(X_train))
    #print("X_train",X_train)
    #print("Y_train",len(Y_train))
    #print("Y_train",Y_train)
    #print("X_test Length",len(X_test))
    #print("X_test",X_test)
    #print("Y_test Length",len(Y_test))
    #print("Y_test",Y_test)
    nb_train_samples = len(X_train)
    nb_val_samples =len(X_test)
    
    new_model=create_model()

    if activation_rnn_size:
       new_model.add(SimpleContext(simple_context, rnn_size, name='simplecontext_1'))

    new_model.add(TimeDistributed(Dense(
            VOCAB_SIZE,
            W_regularizer=regularizer,
            b_regularizer=regularizer,
            name='timedistributed_1')))
    new_model.add(Activation('softmax', name='activation_1'))
    #print("SOFTMAX AND TIMEDISTRIBUTED ADDED TO THE MODEL")
    new_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    K.set_value(new_model.optimizer.lr, np.float32(LR))
    
    #new_model = load_model("/content/gdrive/My Drive/CS_training_E10/FinalWeights/model.h5")
    #new_model.load_weights('/content/gdrive/My Drive/CS_training/EpochWeights/weights-improvement-02-6.95.hdf5')
    load_weights(new_model,'/content/gdrive/My Drive/CS_training_E10/EpochWeights/weights-improvement-09-6.94.hdf5')  
    inspect_model(new_model)
    r = next(gen(X_train, Y_train, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False,  vocab_size=VOCAB_SIZE,  idx2word=index2word))
    traingen = gen(X_train, Y_train, batch_size=batch_size, nb_batches=None, nflips=1, model=new_model, debug=False,  vocab_size=VOCAB_SIZE,  idx2word=index2word)
    valgen = gen(X_test, Y_test, batch_size=batch_size, nb_batches=nb_val_samples // batch_size, nflips=None, model=None, debug=False,  vocab_size=VOCAB_SIZE, idx2word=index2word)
    filepath="/content/gdrive/My Drive/CS_training/CS_Next10_EPOCH/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]
    h = new_model.fit_generator(
        traingen, steps_per_epoch=80, 
        nb_epoch= 9, validation_data=valgen, nb_val_samples=len(X_test),callbacks=callbacks_list
    )
    new_model.save("/content/gdrive/My Drive/CS_training/CSMODEL_Next10/model.h5")
    new_model.save_weights("/content/gdrive/My Drive/CS_training/CSMODEL_Next10/TrainingWeights.hdf5", overwrite=True)
    
    #print("Keras model trained & weights saved in  app/FinalCode/TrainingWeights.hdf5")
    
    # #print samples after training
    gensamples(
        skips=1,
        short=False,
        data=(X_test, Y_test),
        idx2word=index2word,
        vocab_size=VOCAB_SIZE,
        nb_unknown_words=X_test.count(2),
        k=2,          #changed from 10
        oov0=X_test.count(2),       #changed from 10
        batch_size=1, #changed from 2
        temperature=1.0,
        use_unk=False,
        model=new_model,
    )
        

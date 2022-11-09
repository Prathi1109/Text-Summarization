#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Create an LSTM model for  summarization of Research Papers"""
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
#from keras.preprocessing import sequence
#from keras.regularizers import l2
import keras.backend as K

from pre_process import maxlen,CONTENT_SEQ_LEN,DESC_SEQ_LEN,activation_rnn_size,VOCAB_SIZE,w2v
from constants import p_dense,p_emb,p_W, p_U,EMBED_DIMENSION,rnn_layers,rnn_size

 

def create_model():
    
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBED_DIMENSION,embeddings_initializer='uniform',
                        input_length= maxlen , mask_zero=True,dropout=p_emb, weights=[w2v], name = 'embeddingLayer1'))
    model.add(Dense(VOCAB_SIZE, activation='softmax'))
    for i in range(rnn_layers):
        lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                    dropout_W=p_W, dropout_U=p_U,
                    name='lstm_%d'%(i+1))
        model.add(lstm)
        model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))
    
    return model
   
def str_shape(x):
    ###print("STR_SHAPE STARTED")
    return 'x'.join(map(str,x.shape))

def inspect_model(model):
    """##print the structure of Keras `model`."""
    for i, l in enumerate(model.layers):
        ##print(i, 'cls={} name={}'.format(type(l).__name__, l.name))
        weights = l.get_weights()
        print("weights",weights)
        print_str = ''
        for weight in weights:
            print_str += str_shape(weight) + ' '
        ##print(print_str)
        ##print()


class SimpleContext(Lambda):
    """Class to implement `simple_context` method as a Keras layer."""

    def __init__(self, fn, rnn_size, **kwargs):
        """Initialize SimpleContext."""
        self.rnn_size = rnn_size
        super(SimpleContext, self).__init__(fn, **kwargs)
        self.supports_masking = True

        def compute_mask(self, input, input_mask=None):
        
            return input_mask[:, CONTENT_SEQ_LEN:]

    def compute_output_shape(self, input_shape):
        """Get output shape for a given `input_shape`."""
        nb_samples = input_shape[0]   #Prathibha: input_shape[0] : get the number of rows (No of documents here)
        n = 2 * (self.rnn_size - activation_rnn_size)
        return (nb_samples, DESC_SEQ_LEN, n)
    
def simple_context(X, mask, n=activation_rnn_size):
        """Reduce the input just to its headline part (second half).
        For each word in this part it concatenate the output of the previous layer (RNN)
        with a weighted average of the outputs of the description part.
        In this only the last `rnn_size - activation_rnn_size` are used from each output.
        The first `activation_rnn_size` output is used to computer the weights for the averaging.
        """
        ##print("simple_context STARTED")
        ##print("MASK",mask)
        desc, head = X[:, :CONTENT_SEQ_LEN, :], X[:, CONTENT_SEQ_LEN:, :]
        head_activations, head_words = head[:, :, :n], head[:, :, n:]#Prathibha: n is the number of units that are not activated
        ##print( "HEAD_WORDS ",head_words)
        ##print("HEAD_ACTIVATIONS ",head_activations)
        desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]#Prathibha: n is the number of units that are not activated
        ##print("CONTENT_SEQ_LEN ",CONTENT_SEQ_LEN)
        # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        # activation for every head word and every desc word
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
        # make sure we dont use description words that are masked out
        activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :CONTENT_SEQ_LEN], 'float32'), 1)
        ##print("ACTIVATION_ENERGIES", activation_energies)
        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies, (-1, CONTENT_SEQ_LEN))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, DESC_SEQ_LEN, CONTENT_SEQ_LEN))
        ##print("ACTIVATION_WEIGHTS", activation_weights)
        # for every head word compute weighted average of desc words
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
        ##print("DESC_AVG_WORD",desc_avg_word)
        return K.concatenate((desc_avg_word, head_words))

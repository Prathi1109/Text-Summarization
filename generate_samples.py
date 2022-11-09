#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Levenshtein
import numpy as np
import random
from keras.preprocessing import sequence
from pre_process import maxlen,CONTENT_SEQ_LEN,DESC_SEQ_LEN,replaceUNK
from constants import eos,empty

def lpadd(x): 
    """Left (pre) pad a description to CONTENT_SEQ_LEN and then add eos.
    The eos is the input to predicting the first word in the headline
    #Prathibha: If n is not greater than CONTENT_SEQ_LEN then (CONTENT_SEQ_LEN-n) zeros will be added to the prefix of x
    Here x is the list containing the ids of the content words(tokens)
    If n is greater than CONTENT_SEQ_LEN then we cut the list to using x[-CONTENT_SEQ_LEN:] (which gives the CONTENT_SEQ_LEN number of ids from the end) 
    and assign n=CONTENT_SEQ_LEN so that the return stmt changes to x+[eos]
    """
    ##print("LPADD STARTED")
    try:
        assert CONTENT_SEQ_LEN >= 0
    except:
        pass
        #print("Assertion in lpadd")
    if CONTENT_SEQ_LEN == 0:
        return [eos]
    n = len(x)
    if n > CONTENT_SEQ_LEN:
        x = x[-CONTENT_SEQ_LEN:]
        n = CONTENT_SEQ_LEN
    return [empty] * (CONTENT_SEQ_LEN - n) + x + [eos] 


def beamsearch(
        predict, start, k, maxsample, use_unk, empty, temperature, nb_unknown_words,
        vocab_size, model, batch_size, avoid=None, avoid_score=1):
    """Return k samples (beams) and their NLL scores, each sample is a sequence of labels.
   
    All samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples.
    """
    #print("BEAMSEARCH STARTED")
    def sample(energy, n, temperature=temperature):
        """
        Prathibha:
        Here energy=scores,temperature=1.0,n=k=1
        """
        
        
        
        """Sample at most n elements according to their energy."""
        n = min(n, len(energy))
        #print("n",n)
        prb = np.exp(-np.array(energy) / temperature)
        #print("prb",prb)
        res = []
        for i in range(n):
            z = np.sum(prb)
            #print("z",z)
            #print("prb/z",prb/z)
            ##print("np.random.multinomial(1, prb / z, 1)",np.random.multinomial(1, prb / z, 1))
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.  # make sure we select each element only once
        #print("res",res)
        return res

    dead_samples = []
    dead_scores = []
    live_k = 1  # samples that did not yet reached eos
    live_samples = [list(start)] #Prathibha: live_samples = samples of content words till CONTENT_SEQ_LEN and eos length 51
    live_scores = [0]
    #print("LIVE_SAMPLES",live_samples) 
    while live_k:
        # for every possible live sample calc prob for every possible label
        """
        Here predict = keras_rnn_predict
        
        
        """
        probs = predict(live_samples, empty=empty, model=model, batch_size=batch_size)

        # total score for every sample is sum of -log of word prb
        #print("np.array(live_scores)[:, None]",np.array(live_scores)[:, None])
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        #print("CAND_SCORES",cand_scores)
        cand_scores[:, empty] = 1e20
        #print("cand_scores after adding empty",cand_scores)
        if not use_unk:
            for i in range(nb_unknown_words):
                cand_scores[:, vocab_size - 1 - i] = 1e20
        #print("avoid",avoid)
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i, a[n]] += avoid_score
        
        live_scores = list(cand_scores.flatten())
        #print("live_scores",live_scores)
        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        #print("scores = dead_scores + live_scores",scores)
        ranks = sample(scores, k)
        #print("ranks",ranks)
        n = len(dead_scores)
        #print("len(dead_scores",n)
        ranks_dead = [r for r in ranks if r < n]
        ranks_live = [r - n for r in ranks if r >= n]
        #print("ranks_dead",ranks_dead)
        #print("ranks_live",ranks_live)
        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]
        #print("dead_scores",dead_scores)
        #print("dead_samples",dead_samples)
        live_scores = [live_scores[r] for r in ranks_live]
        #print("live_scores",live_scores)
        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        #print("voc_size",voc_size)
        live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_live]
        #print("live_samples",live_samples)
        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of DESC_SEQ_LEN
        zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]
        #print("zombie",zombie)
        # add zombies to the dead
        dead_samples += [s for s, z in zip(live_samples, zombie) if z]
        dead_scores += [s for s, z in zip(live_scores, zombie) if z]
        
        #print("dead_samples += [s for s, z in zip(live_samples, zombie) if z]",dead_samples)
        #print("dead_scores += [s for s, z in zip(live_scores, zombie) if z]",dead_scores)
        
        
        # remove zombies from the living
        live_samples = [s for s, z in zip(live_samples, zombie) if not z]
        live_scores = [s for s, z in zip(live_scores, zombie) if not z]
        live_k = len(live_samples)
        #print("live_samples = [s for s, z in zip(live_samples, zombie) if not z]",live_samples)
        #print("live_scores = [s for s, z in zip(live_scores, zombie) if not z]",live_scores)
        #print("live_k",live_k)
    #print("BEAMSEARCH END") 
    t,d=dead_samples + live_samples, dead_scores + live_scores
    #print("t,d",t,d)
    return dead_samples + live_samples, dead_scores + live_scores


def keras_rnn_predict(samples, empty, model, batch_size):
    """For every sample, calculate probability for every possible label.
    You need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    ##print("KERAS_RNN_PREDICT STARTED")
    #print("KERAS_RNN_PREDICT samples",samples)
    sample_lengths = list(map(len, samples))
    #print("KERAS_RNN_PREDICT sample_lengths",sample_lengths) #Prathibha: map(functiontoapply,list of inputs)
    """
    Prathibha:
    Here we just have one sample since the length of the content is not greater than maxlen. If it is then we need to divide the content
    part into samples and pass
    
    
    """
    
    try:
        assert all(l > CONTENT_SEQ_LEN for l in sample_lengths)
    except:
        pass
        #print("Assertion in rnn_predict sample_lengths")  
    try:    
        assert all(l[CONTENT_SEQ_LEN] == eos for l in samples)
    except:
        pass
        
        #print("Assertion in rnn_predict samples")
        
    # pad from right (post) so the first CONTENT_SEQ_LEN will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    #print("DATA KERAS_RNN_PREDICT", data)      #Prathibha: If words are less than maxlen then append zeros to the right
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    #print("PROBS KERAS_RNN_PREDICT",probs)
    ##print("KERAS_RNN_PREDICT END")
    y=np.array([prob[sample_length - CONTENT_SEQ_LEN - 1]
                     for prob, sample_length in zip(probs, sample_lengths)])
    #print("KERAS_RNN_PREDICT return",y)
    return np.array([prob[sample_length - CONTENT_SEQ_LEN - 1]
                     for prob, sample_length in zip(probs, sample_lengths)])


def vocab_fold(xs, oov0, glove_idx2idx, vocab_size, nb_unknown_words):
    """Convert list of word indices that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    ##print("VOCAB_FOLD STARTED")
    xs = [x if x < oov0 else glove_idx2idx.get(x, x) for x in xs]
    #print("xs",xs)
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= oov0]) #Prathibha: All indices are converted to the glove_idx2idx indices and eos will be added using lpadd
    #print("outside",outside)
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
      
                   
    """
        Prathibha:
        Here oov0 = 60
        so word indices above 60 it adds the indices to the outside list
        In the outside dictionary each element of the list will be a key and the values will be 
        vocab_size - 1 - min(i, nb_unknown_words - 1), nb_unknown_words=10
        Ex: 68-1-0(67),68-1-1(66),68-1-2(65)
    """    
    #print("outside dict",outside)
    xs = [outside.get(x, x) for x in xs]  #Prathibha: xs will have 51 elements together with eos
    #print("xs",xs)
    ##print("VOCAB_FOLD END")
    return xs


def vocab_unfold(desc, xs, oov0):
    """Covert a description to a list of word indices."""
    # assume desc is the unfolded version of the start of xs
    ##print("VOCAB_UNFOLD STARTED")
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    ##print("VOCAB_UNFOLD END")        
    return [unfold.get(x, x) for x in xs]


def gensamples(
        skips, short, data, idx2word, oov0, vocab_size,
        nb_unknown_words, avoid=None, avoid_score=1, **kwargs):
    """Generate text samples."""
    #print("GENSAMPLES STARTED")
    # unpack data
    X, Y = data   #Prathibha: X,Y are the test document content and summary
    #print("gensamples X",X)
    #print("gensamples Y",Y)
    # if data is full dataset pick a random header and description
    if not isinstance(X[0], int):   #Prathibha: If there are many test documents
        i = random.randint(0, len(X) - 1)
        x = X[i]
        y = Y[i]
    else:
        x = X     #Prathibha: If there is just one test document 
        y = Y
       
    #print("y[:DESC_SEQ_LEN]",y[:DESC_SEQ_LEN])
    #print("x[:CONTENT_SEQ_LEN]",x[:CONTENT_SEQ_LEN])
    # #print header and description
    #print('SUMMARY:', ' '.join(idx2word[w] for w in y[:DESC_SEQ_LEN])) #Prathibha: Display the test document Summary which we have
    #print('CONTENT:', ' '.join(idx2word[w] for w in x[:CONTENT_SEQ_LEN])) #Prathibha: Display the test document Content which we input to the model

    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid, str) or isinstance(avoid[0], int):
            avoid[avoid]
        avoid = [a.split() if isinstance(a, str) else a for a in avoid]
        avoid = [[a] for a in avoid]

    print('SUMMARY:')
    samples = []
    if CONTENT_SEQ_LEN == 0:
        skips = [0]
    else:
        skips = range(min(CONTENT_SEQ_LEN, len(x)), max(CONTENT_SEQ_LEN, len(x)), abs(CONTENT_SEQ_LEN - len(x)) // skips + 1) 
    """
    Prathibha:
    If the length of the content of the test document is more than the CONTENT_SEQ_LEN it splits and then process
    #Prints range(start,stop,step) 
    Example range(38,50,13) 
    skips= range(38,50,13)
    for s in skips => for s in 38
    """    
    #print("skips",skips)    
    for s in skips:  
        start = lpadd(x[:s]) #Prathibha: Padding to first s no of words in the content with zeros to match the maxlen
        ##print("start",start)
        fold_start = replaceUNK(start)
        sample, score = beamsearch(
            predict=keras_rnn_predict,
            start=fold_start,
            maxsample=maxlen,
            empty=empty,
            nb_unknown_words=nb_unknown_words,
            vocab_size=vocab_size,
            avoid=avoid,
            **kwargs
        )
        try:
            assert all(s[CONTENT_SEQ_LEN] == eos for s in sample)
        except:
            pass
            #print("Assertion in gensamples")    
        samples += [(s, start, scr) for s, scr in zip(sample, score)]
    #print("SAMPLES",samples)
    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample, oov0)[len(start):]
        #print("sample",sample)
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
            if distance > -0.6:
                print(score, ' '.join(words))
        else:
                print(score, ' '.join(words))
        codes.append(code)
    ##print("CODES",codes) 
    ##print("WORDS",words)
    #print("GENSAMPLES END")    
    return samples

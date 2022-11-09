#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Give the same path in word2vec.load and embeddingModel.save functions (line 130 and 134)
change the path in the createEmbeddingModel function to the same file path where our dataset is present which we gave in the training.py program


"""

import nltk
from nltk.stem import WordNetLemmatizer
import string
import json
global graph,model
from constants import empty,eos,unk,numOfGrams,SLIDING_WINDOW_SIZE,gramsFilePath,EMBED_DIMENSION
import re
import pickle
from gensim.models import Word2Vec
import time
from os import listdir
from collections import Counter
import numpy as np
from itertools import tee

wordnetlemmatizer=WordNetLemmatizer()

index2index = {}
word2index = {}
index2word = {}
VOCAB_SIZE = 0
#w2v = []
maxlen = 100
content_AllDocs = []
desc_AllDocs = []



def buildNGrams(tokens):

    ngrams = list(nltk.ngrams(tokens, numOfGrams))
    return ngrams; 

def preprocess(text):
    
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text) #regex to remove urls
    
    text = re.sub(r'\d', '', text)#regex to remove digits
    
    tokens= nltk.word_tokenize(text)

    preprocessedWords = []
    for word in tokens:
        #if word not in setOfStopWords:
         updatedWord = wordnetlemmatizer.lemmatize(word)
         if updatedWord not in string.punctuation:
            preprocessedWords.append(updatedWord)
    
    grams = buildNGrams(preprocessedWords)
    return preprocessedWords,grams;

def createEmbeddingModel(domainFolderPath, CONTENT_SEQ_LEN,DESC_SEQ_LEN,index2index = index2index, index2word = index2word, word2index = word2index):
    

    word2index['<empty>'] = empty
    index2word[empty] = '<empty>'
    word2index['<eos>'] = eos
    index2word[eos] = '<eos>'
    word2index['<unk>'] = unk
    index2word[2] = '<unk>'
    w2v = []
    #currentDomain = domainFolderPath.split("//")[-2]
    firstDocFlag = True
    
    files = list(listdir(domainFolderPath))
    print("files",files)
    ##print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print("###files###",files)
    gramsIndicesList = []
    for filePath in files:
        gramsList = []
        DomainVocab = []
        gramsDict = {}
        print("domainFolderPath+filePath",domainFolderPath+filePath)
        with open(domainFolderPath+filePath,"r",encoding="utf-8") as fin:
            startTime = time.process_time()
            jsonLines=list(fin.readlines())
            print("JSONLINES", len(jsonLines))
            #print(filePath + " reading successful - length - "+str(len(jsonLines)))
            #EMBED_DIMENSION = 100
            contentList = []
            descList = []
            for l in range(len(jsonLines)):
                doc=json.loads(jsonLines[l])
                contentAndConclusion = doc["content"] + " " + doc["conclusion"]
                description = doc["description"]
                desc,descGrams = preprocess(description.lower())
                content,contentGrams = preprocess(contentAndConclusion.lower())
                contentList.append(content)
                descList.append(desc)
                gramsList.extend(descGrams)
                gramsList.extend(contentGrams)
                #CONTENT_SEQ_LEN = max(len(content),CONTENT_SEQ_LEN) 
                #DESC_SEQ_LEN = max(len(desc),DESC_SEQ_LEN)
                
                docWords = []
                docWords.extend(content)
                docWords.extend(desc)
                DomainVocab.append(docWords)
                del docWords
#                #print("len((DomainVocab)",len())
          
                    #print(str(l)+" lines have been processed so far.")
            #print("contentList",len(contentList) )   
            
            #print(filePath + " pre-processing successful")
            
#            #print("DomainVocab",DomainVocab)
            
            if firstDocFlag:
                #print(filePath + " is the first document to be processed")
                embeddingModel = Word2Vec(DomainVocab,min_count=30)
                embeddingModel.train(DomainVocab,total_examples=len(DomainVocab),epochs=3)
                #embeddingModel.save(constants.dataPath+currentDomain+'_Embeddings.model')
 
                firstDocFlag = False
            else:
                #print(filePath + " is NOT the first document to be processed")
                embeddingModel = Word2Vec.load("/content/gdrive/My Drive/CS_training/DomainInfo_Next10/Test_Embeddings.model")
                embeddingModel.build_vocab(DomainVocab, update=True)
                embeddingModel.train(DomainVocab,total_examples=len(DomainVocab),epochs=3)#,min_count=30)
            
            embeddingModel.save('/content/gdrive/My Drive/CS_training/DomainInfo_Next10/Test_Embeddings.model')
            VOCAB_SIZE=len(embeddingModel.wv.vocab)+5
            w2v = np.zeros((VOCAB_SIZE, EMBED_DIMENSION))
            for index in range(len(embeddingModel.wv.vocab)):
                currentIndex = index + 3
                word = embeddingModel.wv.index2word[index]
                w2v[index] = embeddingModel.wv[word]
                word2index[word] = currentIndex
                index2word[currentIndex] = word
                #index2index[index] = currentIndex #<created word-embedding index> : <our model index> #to check key value order
            ##print("############################word2index##############",word2index)
            del DomainVocab
            
            gramsDict = Counter(gramsList)
            
            ##print(gramsDict)
            ##print("Before deleting oov bigrams & converting into indices: ",len(gramsDict))
            for gd in gramsDict.keys():
                tmp = list(gd)
                indexTmp = []
                counter = 0
                for t in tmp:
                    if t not in embeddingModel.wv.vocab:
                        break
                    else:
                        counter = counter + 1
                        indexTmp.append(word2index[t])
                if counter == 2:
                    gramsIndicesList.append(tuple(indexTmp))
            
            
            del gramsList
            del gramsDict
                
            ##print("CONTENT_SEQ_LEN: ",CONTENT_SEQ_LEN) 
            ##print("DESC_SEQ_LEN",DESC_SEQ_LEN)
            
            #print(filePath + " embeddings created.")
            timeTaken = time.process_time() - startTime
            #print("Time taken for "+filePath+" embeddings: "+str(timeTaken))
            
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    

                        

    gramsIndicesDict = Counter(gramsIndicesList)
    del gramsIndicesList
    ##print("Length After deleting oov bigrams: ",len(gramsIndicesDict))
    
    #Writing the bigrams into a file 
    gramsFilePath = "/content/gdrive/My Drive/CS_training/DomainInfo_Next10/gramsDict.txt"
    with open(gramsFilePath, 'wb') as pickleWriter:
        pickle.dump(gramsIndicesDict, pickleWriter)
    del gramsIndicesDict
    #Adding eos token to both 
    #CONTENT_SEQ_LEN = CONTENT_SEQ_LEN + 1
    #DESC_SEQ_LEN = DESC_SEQ_LEN + 1
    CONTENT_SEQ_LEN = 900
    DESC_SEQ_LEN = 100
    maxlen = CONTENT_SEQ_LEN + DESC_SEQ_LEN 
    #print("Max Seq length ",maxlen)
    vocabInfoPath="/content/gdrive/My Drive/CS_training/DomainInfo_Next10/vocab_info.txt"
    with open(vocabInfoPath, 'wb') as pickleWriter:
        pickle.dump((word2index,index2word,index2index,w2v),pickleWriter,2)


    return CONTENT_SEQ_LEN, DESC_SEQ_LEN,maxlen ,embeddingModel,contentList,descList,w2v,word2index,index2word;

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters);

def replaceUNK(sequenceOfIndices):

    #unk = constants.unk
    #windowSize = constants.SLIDING_WINDOW_SIZE
    gramsFilePath = "/content/gdrive/My Drive/CS_training/DomainInfo_Next10/gramsDict.txt"
    with open(gramsFilePath, 'rb') as pickleReader:
        gramsDict = pickle.loads(pickleReader.read())

    if len(gramsDict) > 0:
        for w in window(range(len(sequenceOfIndices)), SLIDING_WINDOW_SIZE):
            tmp = list(w)
            
            currentWindow = [sequenceOfIndices[wi] for  wi in tmp]
            query = []
            numOfUNK = currentWindow.count(2)
            ##print("This window has - ",numOfUNK)
            if numOfUNK >= 2: #if more than 2 unk tokens in a window of size 3
    
                if (currentWindow[0] == unk and currentWindow[2] == unk):
                    query.append(tmp[0:2])
    
            elif numOfUNK == 1:
    
                if currentWindow[0] == unk:
                    query.append(tmp[0:2])
    
                elif currentWindow[1] == unk:
                    query.append(tmp[0:2])
                    query.append(tmp[1:3])
                else:
                    query.append(tmp[1:3])
            else:
                continue
            
            if len(query) == 1:
                
                ##print("query: ",query)
                #eg = [sequenceOfIndices[q] for q in query[0]]
                
                keywordIndex = -1
                unkIndex = -1
                
                temp = []
                temp.extend(query[0])
                for q in temp:
                   
                    if sequenceOfIndices[q] != unk:
                        keywordIndex = q
                    else:
                        unkIndex = q
    
                counter = 0 
                if keywordIndex != -1:
                    for j in gramsDict.keys():
                        keywordPosInTuple = temp.index(keywordIndex)
                        if j[keywordPosInTuple] == sequenceOfIndices[keywordIndex]:
                            if max(counter,gramsDict[j]) == gramsDict[j]: #most frequent bigram chosen
                                #print("j: ",j)
                                counter = gramsDict[j]
                                if keywordPosInTuple == 0:
                                    sequenceOfIndices[unkIndex] = j[1]
                                else:
                                    sequenceOfIndices[unkIndex] = j[0]
                            else:
                                continue
                        else:
                            continue
                else:
                    continue
                
            elif len(query) == 2:
    
                keywordIndices = [-1,-1]
                assert query[0][1] == query[1][0]
                unkIndex = query[0][1]
                keywordIndices[0] = query[0][0]
                keywordIndices[1] = query[1][1]
				
				
                if keywordIndices.count(-1) == 0:
                    for j in gramsDict.keys():
                        if j[0] == sequenceOfIndices[keywordIndices[0]]:
                            for k in gramsDict.keys():
                                if j != k and j[1] == k[0]:
                                    if k[1] == sequenceOfIndices[keywordIndices[1]]:
                                        sequenceOfIndices[unkIndex] = k[0]
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            continue
                else:
                    continue

    ##print("Final output: ",embeddingOutput)
    ##print("Final output: ",sequenceOfIndices)
    ##print("*************************************************")
    return sequenceOfIndices;         


CONTENT_SEQ_LEN,DESC_SEQ_LEN,maxlen,embedModel,contentlist,desclist,w2v,word2index,index2word= createEmbeddingModel("/content/gdrive/My Drive/CS_training/data/DataSet/CStrain/",0,0)
VOCAB_SIZE=len(embedModel.wv.vocab)+5
print("VOCAB_SIZE",VOCAB_SIZE)
#print(w2v)
#VOCAB_SIZE=VOCAB_SIZE+1
#embw2v=w2v.values()


activation_rnn_size = 25 if CONTENT_SEQ_LEN else 0

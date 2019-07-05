# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk
"""

'''
Notes:
    -- first version extracted from dnaNet_v7
    -- contains only code for convolutional models
    -- OBS: not sure that the code runs as is (been a long time since I ran it). So undoubtedly needs some nursing ....
   
Usage:
    
In general: all functions called "allInOne"-something include/call all what is needed for training/validation fo a particular model.
So allInOneWithDynSampling_ConvModel_II, will train/test a convo model; the function calls the code for building the model, for compiling 
it and for dynamically sampling from the desired data. The parameters of the function allow to specify the model, the sampling and 
the training.
    

####################################################

Input data:

####################################################

# Human genome 

#rootGenome = r"/Users/newUser/Documents/clouds/Sync/Bioinformatics/various_python/DNA_proj/data/human/"

#On binf servers:
rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName



####################################################

#Set up training schedule, model and run:

####################################################

#for short test run (eg to see how long training takes)
nrOuterLoops = 2
nrOfRepeats = 2
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 3
batchSize = 100
stepsPerEpoch = 500
trainDataIntervalStepSize = 2000000
trainDataInterval = [0,5000000]
nrTestSamples = 100000 
testDataInterval = [10000000,10500000]


#In anger:
nrOuterLoops = 1
nrOfRepeats = 100
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [0,0]


#A small test model
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [3, 3]
nrFilters = [64, 64] 
padding = 'valid'
#Final dense layers:
hiddenUnits = [50] 
onlyOneRandomChromo_b = 0
avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1
str = '2LayersFlat3_1Dense50_learningRate001_padValid_noPool_augWithCompl'
modelName = 'ownSamples/human/inclRepeats/Conv1d_' + str + '_50_units1Hidden'  
modelDescr = str+'_50_units1Hidden'



#The larger model
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [3, 3,  6, 6, 9, 9]
nrFilters = [64, 64, 96, 96, 128, 128] 
padding = 'valid'
sizeOutput=4
#final dense layers:
hiddenUnits = [100, 50]
onlyOneRandomChromo_b = 0
#??
avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1
str = '6LayersRising3To9_2Dense100_50_learningRate001_padValid_noPool_augWithCompl'
modelName = 'ownSamples/human/inclRepeats/Conv1d_' + str + '_50_units1Hidden'  
modelDescr = str+'_50_units1Hidden'



#Ex/Including results from frq model:
dynSamplesTransformStyle_b = 0
inclFrqModel_b = 0
insertFrqModel_b = 0
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
#file = "frqModel_chr10_k4.txt"
file = "frqModel_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
exclFrqModelFlanks_b = 0
augmentWithRevComplementary_b = 1
dropoutVal = 0.02 #corr's to 1 unit
dropoutLastLayer_b = 1
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.1 #default, but we use Adam here, so the value here isn't used
learningRate = learningRate

#Run training/testing:
dnaNet.allInOneWithDynSampling_ConvModel_I(nrOuterLoops = nrOuterLoops, firstIterNr = 0, nrOfRepeats = nrOfRepeats, firstRepeatNr = 0, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval0 = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

   

'''



#THEANO_FLAGS='floatX=float32,device=cuda' 
#TENSORFLOW_FLAGS='floatX=float32,device=cuda' 

import os

#set this manually at very beginning of python session (dont set anything when using Hecaton; weill get you the GTX 1080, which has high enough compute capability)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#to prevent the process from taking up all the ram on the gpu upon start:
#import tensorflow as tf
import tensorflow as tf

config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

#tf.device('/gpu:1')

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras




from keras import utils, backend

from keras.models import Sequential, Model

#Conv1D
from keras.layers import Conv1D, Conv2D, Input, Dense, Dropout, AveragePooling1D, MaxPooling1D, AveragePooling2D, MaxPooling2D, Flatten, Concatenate, Reshape, merge
#Additional for LSTM
from keras.layers import LSTM, Activation, Bidirectional, concatenate, Lambda, multiply, Add, RepeatVector, Permute, Dot

from keras.optimizers import SGD, RMSprop, Adam
from keras.models import model_from_json

from keras.utils.vis_utils import plot_model

from scipy.fftpack import fft, ifft

import numpy as np
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from random import shuffle

import frqModels as frqM

import cPickle as pickle

#import graphviz
#import pydot



from dnaNet_dataGen import * #all smpling aso is here


############################## 
##        Conv's
##############################

def nrOfParametersConv(filters, sizeOfFilters, nrOfChannels):
    ''' Computes nr of parameters in densely connected 
    convolutional network.'''
    
    if len(filters) > len(sizeOfFilters):
        print("nr of filters cannot be larger than the number of the sizeOfFilters list")
        return -1
    
    nrOfParams = filters[0]*sizeOfFilters[0]*nrOfChannels
    
    for i in range(1,len(filters)):
        
        nrOfParams += filters[i]*filters[i-1]*sizeOfFilters[i]*nrOfChannels

    return nrOfParams

def makeConv1Dmodel(sequenceLength, 
                    letterShape, 
                    lengthWindows, 
                    nrFilters, 
                    hiddenUnits = [50], 
                    outputSize = 4, 
                    padding = 'same',
                    pool_b = 0, 
                    maxPooling_b = 1, 
                    poolStrides = 1,
                    poolAt = [], 
                    dropoutVal = 0.25,
                    dropoutLastLayer_b = 0):
    '''
    Builds 1D convolutional network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the last flt layer
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv1d model...')    
    
    inputs   = Input(shape=(sequenceLength,letterShape))
    #conv    = Conv1D(kernel_size=window, strides=1, filters=nrFilters[0], padding='valid', activation='relu')(inputs)
    conv    = Conv1D(kernel_size=lengthWindows[0], strides=1, filters=nrFilters[0], padding=padding, activation='relu')(inputs)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            pool = MaxPooling1D(strides = poolStrides)(conv)
        else:
            pool = AveragePooling1D(strides = poolStrides)(conv)
    else:
        pool = conv
#    pool    =  MaxPooling1D()(conv)
    
    for i in range(1,len(nrFilters)):    
    
        newConv = Conv1D(kernel_size=lengthWindows[i], strides=1, filters=nrFilters[i], padding=padding, activation='relu')(pool)
        
        if pool_b == 1  and poolAt.count(i) > 0:
            if maxPooling_b == 1:
                newPool = MaxPooling1D(strides = poolStrides)(newConv)
            else:
                newPool = AveragePooling1D(strides = poolStrides)(newConv)
        else:
            newPool = newConv  
       
        conv = newConv
        pool = newPool


    output = Flatten()(pool)

    nrHiddenLayers = len(hiddenUnits)
    for i in range(0, nrHiddenLayers-1):    
        
        #pre 16 Aug 18:
#        features = Dropout(dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(output))
#        #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))
#        output   = Dense(outputSize, activation='softmax')(features)     
#    
        #post 16 aug 18:
        output = Dropout(dropoutVal)(Dense(hiddenUnits[i], activation='relu')(output))
        
        
    
#    flatten  = Flatten()(pool) #1st: (pool6) 2nd: (pool3) 3rd: (pool2)
    features = Dropout(dropoutLastLayer_b*dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(output))
    #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))
    output   = Dense(outputSize, activation='softmax')(features)
    model    = Model(inputs=inputs, outputs=output)
    
    return model


def makeConv1DmodelMergedWithFrqModel(frqModelOutputSize,
                                     sequenceLength, 
                    letterShape, 
                    lengthWindows, 
                    nrFilters, 
                    hiddenUnits = [100], 
                    outputSize = 4, 
                    padding = 'same',
                    pool_b = 0, 
                    maxPooling_b = 1, 
                    poolStrides = 1,
                    poolAt = [],
                    dropoutVal = 0.25,
                    dropoutLastLayer_b = 0,
                    mergeFrqAndConvInLastDense_b = 0):
    '''
    Builds 1D convolutional network model merged with the freqeuncy model; the input is divided in a 
    sequenece for the convolutional part and the output from the frq model; the two are input to the
    dense layer following the conv model:
    
    orig input 
    
    --> 1) two flanks of size flankSize - flankSizeFrqModel goes two conv layers
    --> 2) midle two flanks of size flankSizeFrqModel goes to frq model
    
    The output from 1 and 2 are merged in the (first) dense layer. So in effect the input's middle word
    (skipping the central letter) of length 2*flankSizeFrqModel are skipping the conv layers and instead
    handled by the frq model.
    
    sequenceLength = lenght of the sequence input to conv layers (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the conv layers (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each conv layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the last flt layer
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv1d model merged with frq model output ...')    
    
    inputsFrqModel = Input(shape=(frqModelOutputSize,letterShape))
    inputs   = Input(shape=(sequenceLength,letterShape))
    #conv    = Conv1D(kernel_size=window, strides=1, filters=nrFilters[0], padding='valid', activation='relu')(inputs)
    conv    = Conv1D(kernel_size=lengthWindows[0], strides=1, filters=nrFilters[0], padding=padding, activation='relu')(inputs)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            pool = MaxPooling1D(strides = poolStrides)(conv)
        else:
            pool = AveragePooling1D(strides = poolStrides)(conv)
    else:
        pool = conv
#    pool    =  MaxPooling1D()(conv)
    
    for i in range(1,len(nrFilters)):    
    
        newConv = Conv1D(kernel_size=lengthWindows[i], strides=1, filters=nrFilters[i], padding=padding, activation='relu')(pool)
        
        if pool_b == 1  and poolAt.count(i) > 0:
            if maxPooling_b == 1:
                newPool = MaxPooling1D(strides = poolStrides)(newConv)
            else:
                newPool = AveragePooling1D(strides = poolStrides)(newConv)
        else:
            newPool = newConv  
       
        conv = newConv
        pool = newPool    

    
#    flattenConvOutput  = Flatten()(pool) #1st: (pool6) 2nd: (pool3) 3rd: (pool2)
    
#    print "pool shape ", backend.shape(pool)

    #"flatten" the output from conv before passing it to the final densely connected std NN layers:
    output = Reshape((-1,))(pool)
    
    nrHiddenLayers = len(hiddenUnits)
    print("The nrHiddenLayers is: ", nrHiddenLayers)
    for i in range(0, nrHiddenLayers-1):    
        
        #pre 16 Aug 18:
#        features = Dropout(dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(output))
#        #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))
#        output   = Dense(outputSize, activation='softmax')(features)     
#    
        #post 16 aug 18:
        output = Dropout(dropoutVal)(Dense(hiddenUnits[i], activation='relu')(output))
        
    #    inputsFrqModel = Input(shape=(frqModelOutputSize,letterShape))
#    print "inputsFrqModel shape ", backend.shape(inputsFrqModel)

    if mergeFrqAndConvInLastDense_b == 1:
        
        #for the last NN layer: merge with frq model output with the output from the conv+NN-layers:
    #    pool = Reshape((pool._keras_shape[1]*pool._keras_shape[2],))(pool)
    #    print(reshapedPool._keras_shape)    
        reshapedInputsFrqModel = Reshape((-1,))(inputsFrqModel)
        print(reshapedInputsFrqModel._keras_shape)
    
        merge_outputs = Concatenate()([reshapedInputsFrqModel, output])
        print(merge_outputs._keras_shape)
        
    #    flattenFrqModelOutput = Flatten()(inputsFrqModel)
    #    merge_outputs = Concatenate([flattenConvOutput, flattenFrqModelOutput], axis = -1)
        flatten = Reshape((-1,))(merge_outputs)
        print(flatten._keras_shape)
    
#    flatten = Flatten()(merge_outputs)
#    flatten = Flatten()(pool)
    
    else:
        
        flatten = output
    
    
    features = Dropout(dropoutLastLayer_b*dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(flatten))
    #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))

    if mergeFrqAndConvInLastDense_b != 1:
        
        #for the final decision layer: merge with frq model output with the output from the conv+NN-layers:
    #    pool = Reshape((pool._keras_shape[1]*pool._keras_shape[2],))(pool)
    #    print(reshapedPool._keras_shape)    
        reshapedInputsFrqModel = Reshape((-1,))(inputsFrqModel)
        print(reshapedInputsFrqModel._keras_shape)
    
        merge_outputs = Concatenate()([reshapedInputsFrqModel, features])
        print(merge_outputs._keras_shape)
        
    #    flattenFrqModelOutput = Flatten()(inputsFrqModel)
    #    merge_outputs = Concatenate([flattenConvOutput, flattenFrqModelOutput], axis = -1)
        features = Reshape((-1,))(merge_outputs)
        print(features._keras_shape)


    output   = Dense(outputSize, activation='softmax')(features)
    model    = Model(inputs=[inputsFrqModel, inputs], outputs=output)
    
    return model


def makeConv2Dmodel(sequenceLength, 
                    letterShape, 
                    lengthWindows, 
                    nrFilters, 
                    strides = (1,1),
                    hiddenUnits = [100], 
                    outputSize = 4, 
                    padding = 'same',
                    pool_b = 0, 
                    maxPooling_b = 0, 
                    poolStrides = (1,1),
                    poolAt = [], 
                    dropoutVal = 0.25,
                    dropoutLastLayer_b = 0):
    '''
    Builds 2D convolutional network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the last flt layer
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv2d model...')    
    
    inputs = Input(shape=(sequenceLength,sequenceLength,letterShape*letterShape))

    conv = Conv2D(data_format="channels_last", kernel_size=lengthWindows[0], strides=strides, filters=nrFilters[0], padding=padding, activation='relu')(inputs)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            pool = MaxPooling2D(strides = poolStrides)(conv)
        else:
            pool = AveragePooling2D(strides = poolStrides)(conv)
    else:
        pool = conv
#    pool    =  MaxPooling1D()(conv)
    
    for i in range(len(nrFilters)-1):    
    
        newConv    = Conv2D(data_format="channels_last", kernel_size=lengthWindows[i+1], strides=strides, filters=nrFilters[i+1], padding=padding, activation='relu')(pool)
        
        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                newPool = MaxPooling2D(strides = poolStrides)(newConv)
            else:
                newPool = AveragePooling2D(strides = poolStrides)(newConv)
        else:
            newPool = newConv  
       
        conv = newConv
        pool = newPool


    output = Flatten()(pool)
    
    nrHiddenLayers = len(hiddenUnits)
    for i in range(0, nrHiddenLayers-1):    
        
        features = Dropout(dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(output))
        #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))
        output   = Dense(outputSize, activation='softmax')(features)     
        
    
#    flatten  = Flatten()(pool) #1st: (pool6) 2nd: (pool3) 3rd: (pool2)
    features = Dropout(dropoutLastLayer_b*dropoutVal)(Dense(hiddenUnits[nrHiddenLayers-1], activation='relu')(output))
    #features = Dropout(0.25)(Dense(sizeHidden, activation='relu')(flatten))
    output   = Dense(outputSize, activation='softmax')(features)
    model    = Model(inputs=inputs, outputs=output)
    
    return model


#All in one run
#labelsCodetype NOT ENCORPORATED
def allInOne_ConvModel(bigLoopsNr = 1,
                       startFrom = 0,
            loadModelFromFile_b = 0,
            modelIs1D_b = 1, 
            loss = "categorical_crossentropy", 
            learningRate = 0.025,
            momentum = 0.001,
            nrTrainSamples = 100000,
            trainDataInterval = [0,200000] , 
            nrValSamples = 20000,
            valDataInterval = [200000,400000],   
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000], 
            customFlankSize_b = 0, 
            customFlankSize = 50,
            genSamples_b = 0, 
            genomeFileName = '',
            outputEncodedOneHot_b = 1,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
            genSamplesFromRandomGenome_b = 0, 
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            augmentWithRevComplementary_b = 0, 
            batchSize = 128, 
            nrEpochs = 100,
            sizeOutput=4,
            letterShape = 4, # size of the word
            lengthWindows = [2, 3, 4, 4, 5, 10],
            hiddenUnits = [50],
            dropoutVal = 0.25,
            dropoutLastLayer_b = 0,
            nrFilters = [50, 50, 50, 50, 50, 50],    
            padding = 'same',  
            pool_b = 0,
            maxPooling_b = 0,
            poolAt = [],
            poolStrides = 1,
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'Conv type ... to be filled in!',
            on_binf_b = 1):
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


            
#1st version parameters:    
#    lengthWindows = 5, 7, 10, 15, 20, 25
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 30, 20, 10
    
#2nd version parameter:
#    lengthWindows = 3, 5, 7
#    nrHiddenUnits = 25
#    nrFilters = 25, 15, 10
      
##3rd version parameter:
#    lengthWindows = 3, 3
#    nrHiddenUnits = 25
#    nrFilters = 10, 5
     
#    lengthWindows = 3, 3, 4, 6, 8, 10, 15, 20, 30
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 45, 40, 30, 25, 20, 15
     
     
    
    #Using the getData-fct to fetch data:
#    fname=root + r"trShort.dat"
#    tname=root + r"tsShort.dat"
#    vname=root + r"vlShort.dat"
#      
#    X,Y = dnaNet.getData2(fname, letterShape, sizeOutput, outputType=float)  
#    sizeInputput = X.shape[1] 
#        
#    Xt,Yt = dnaNet.getData2(tname, letterShape, sizeOutput , outputType=float)  
#    
#   
    
    modelFileName = rootOutput + modelName
    
    for i in range(startFrom, bigLoopsNr):
        
        print("Now at big loop iter: %d " % i)
        
            
        #If the model is 2d we convert all data to images: seq of consisting of two flanks
        #becomes a flankSize-by-flankSize matrix of 4-by-4 matrices, the latter being 
        #the one-hot encoding of the 16 possible letter pairs 
        convertToPict_b = 0
        if modelIs1D_b == 0: #model is 2D
    
            convertToPict_b = 1
        
        
        if genSamples_b > 0.5: #generate a set of random data acc to the sizes set
    
            #if a genomeFileName is specified, use that genome:
            if len(genomeFileName) > 0:
                fromGenome_b = 1
            else:
                fromGenome_b = 0
          
            X,Y, genomeSeqSourceTrain = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTrainSamples, startAtPosition = trainDataInterval[0], endAtPosition = trainDataInterval[1],  convertToPict_b = convertToPict_b,  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
            sizeInput = X.shape[1]
            print("Train data shape", X.shape)
    
            avoidChromo.append(genomeSeqSourceTrain) #to avoid getting val data from the same chromo as the training data 
            Xv,Yv, genomeSeqSourceVal = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrValSamples, startAtPosition = valDataInterval[0], endAtPosition = valDataInterval[1],  convertToPict_b = convertToPict_b,  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
            print("Val data shape", Xv.shape)
            
            avoidChromo.append(genomeSeqSourceVal) ##to avoid getting test data from the same chromo as the training and validation data 
            Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  convertToPict_b = convertToPict_b,  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
            print("Test data shape", Xt.shape)
    
        elif genSamplesFromRandomGenome_b > 0.5: #generate a set of random data acc to the sizes set
    
            #generate random genome of set size:   
            genRandomGenome(length = randomGenomeSize, fileName = root + randomGenomeFileName, on_binf_b = on_binf_b) #will write the generated genome sequence to the file  
    
            X,Y = genSamples_I(nrSamples = trainDataInterval[1] - trainDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
            sizeInput = X.shape[1]
    
            Xv,Yv = genSamples_I(nrSamples = valDataInterval[1] - valDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
    
            Xt,Yt = genSamples_I(nrSamples = testDataInterval[1] - testDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
            
    
        else: #fetch the data from an appropriate source
    
            #Using the getData2-fct to fetch data:  
            fname=root + r"training.dat"
            vname = root + r"validation.dat"
            tname=root + r"test.dat"
        
            
            X,Y = getData2(fname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = trainDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
            sizeInput = X.shape[1]
                    
            Xv,Yv = getData2(vname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = valDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)      
                
            Xt,Yt = getData2(tname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
        
        
        batch_size = min(batchSize,max(1,len(X)/20))
        #print batch_size, nrEpochs
         
         

        #If at first iter of bigLoop build the model, else just reload it
        if i == 0 and loadModelFromFile_b == 0: 
            if modelIs1D_b == 1: 
                net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b)
        
            else:
                
                net = makeConv2Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = (poolStrides, poolStrides), maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b)
        else:
            
            #reload the model from previous iter            
            net = model_from_json(open(modelFileName).read())
            net.load_weights(modelFileName +'.h5')

            print("I've now reloaded the model at, iter: %d", i)
    
    #    utils.plot_model(net, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
        learningRate = learningRate  - i*0.001
        momentum = momentum - i*0.0001       
        sgd = SGD(lr= learningRate, decay=1e-6, momentum= momentum, nesterov=True)
        #sgd = SGD(lr=0.01)
        
        net.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
        
        #net.compile(loss='categorical_crossentropy', optimizer='Adam')
        #net.compile(loss='binary_crossentropy', optimizer=sgd)
        #xor.compile(loss="hinge", optimizer=sgd)
        #xor.compile(loss="binary_crossentropy", optimizer=sgd)
          
        
        #net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, show_accuracy=True, verbose=0)
        history = net.fit(X, Y, batch_size=batch_size, epochs = nrEpochs, verbose=1, validation_data=(Xv,Yv) )
    
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(rootOutput + modelName + '_training_validation_acc_vs_epoch' + '_bigLoopIter' + str(i) + '.pdf')
    #    plt.show()
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(rootOutput + modelName + '_training_validation_loss_vs_epoch' + '_bigLoopIter' + str(i) + '.pdf')
    #    plt.show()
    
        #test it:
        score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
        print('Test score:', score)
        print('Test accuracy:', acc)
        
        if save_model_b == 1:
             
            json_string = net.to_json()
            open(modelFileName, 'w').write(json_string)
            net.save_weights(modelFileName + '.h5',overwrite=True)
            
            
            
        #Write run-data to txt-file for documentation of the run:
        runDataFileName = rootOutput + modelName + '_runData' + '_bigLoopIter' + str(i) + '.txt'
        runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
        
        s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
        s += modelDescription  + "\n"   
        s += 'Performance on test set, loss and accuracy resp.: ' + str(score) + ' ' + str(acc) + "\n"   
        if save_model_b == 1:
            s+= 'Model data obtained after training the model are recorded in: ' +  rootOutput + modelName + ' and ' + rootOutput + modelName + '.h5\n' 
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #Which genome data were used?:
        if genSamples_b > 0.5:
            s = "Samples generated with python code from real genome." + "\n"
            s += "Genome data in file: " + genomeFileName + "\n"
            s += "Letters are one-hot encoded" + "\n"
            s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
            s += "trainDataInterval:" + str(trainDataInterval)  + "\n" 
            s += "nrValSamples:" + str(nrValSamples)  + "\n"
            s += "valDataInterval:" + str(valDataInterval)  + "\n"   
            s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
            s += "testDataInterval:" + str(testDataInterval)  + "\n" 
            if onlyOneRandomChromo_b == 1:
                s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
                s += "Validation data from chromosome: " + genomeSeqSourceVal  + "\n"
                s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
                s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
            else:
                s += "Read in the whole genome sequence" + "\n"
            s += "shuffle_b = " + str(shuffle_b) + "\n"
            s += "inner_b = " + str(inner_b) + "\n"
            s += "shuffleLength = " + str(shuffleLength) +  "\n"
            s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
            s += "trainDataInterval:" + str(trainDataInterval)  + "\n" 
            s += "nrValSamples:" + str(nrValSamples)  + "\n"
            s += "valDataInterval:" + str(valDataInterval)  + "\n"   
            s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
            s += "testDataInterval:" + str(testDataInterval)  + "\n" 
         
        elif genSamplesFromRandomGenome_b > 0.5:
            
            s = "Samples from random genome, all generated with python code." + "\n"
            s += "Genome data in file: " + randomGenomeFileName
        
        else: #fetch the data from an appropriate source
    
            s = "Pre-generated samples (ie not generated with the python code.)" + "\n"
            s += "Training samples from: " + fname  + "\n"
            s += "Validation samples from: " + vname  + "\n"
            s += "Test samples from: " + tname  + "\n"
            s += "shuffle_b = " + str(shuffle_b) + "\n"
            s += "inner_b = " + str(inner_b) + "\n"
            s += "shuffleLength = " + str(shuffleLength) +  "\n"
            
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #various params:    
        s= 'loss = "categorical_crossentropy"\n' 
        s += 'trainDataInterval: ' + str(trainDataInterval) + "\n"
        s += 'valDataInterval: ' + str(valDataInterval) + "\n"
        s += 'testDataInterval: ' + str(testDataInterval) + "\n" 
        s += 'customFlankSize_b: ' + str(customFlankSize_b) + "\n" 
        s += 'customFlankSize: ' + str(customFlankSize) + "\n" 
        s += 'genSamples_b: ' + str(genSamples_b) + "\n" 
        s += 'genomeFileName: ' + genomeFileName + "\n" 
        s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
        s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
        s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n" 
        s += "avoidChromo: " + str(avoidChromo)  + "\n" 
        s += 'genSamplesFromRandomGenome_b: ' + str(genSamplesFromRandomGenome_b) + "\n" 
        s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
        s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
        s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
        s += 'learning rate: ' + str(learningRate)  + "\n" 
        s += 'momentum: ' + str(momentum) + "\n" 
        s += 'batchSize: ' + str(batchSize) + "\n"
        s += 'padding: ' + padding + "\n"
        s += 'pool_b: ' +  str(pool_b) + "\n"
        s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
        s += 'poolAt: ' +  str(poolAt) + "\n"
        s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
        s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
        s += 'letterShape: ' + str(letterShape) + "\n" 
        s += 'save_model_b: ' + str(save_model_b) + "\n" 
        s += 'modelName: ' + modelName + "\n" 
        s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
        
        runDataFile.write(s)
            
        #Params for net:
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        s = 'lengthWindows: ' + str(lengthWindows)  + "\n" 
        s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
        s += 'nrFilters: ' + str(nrFilters)
        
        runDataFile.write(s)
        
        runDataFile.close()
        

#NEEDS REPAIR TO WORK WITH REPEAT NR AND LABELCODETYPE
def allInOneWithDynSampling_ConvModel_I_testOnly(nrOuterLoops = 1,
                                        firstIterNr = 0,
            loadModelFromFile_b = 0,
            modelFileName = '',
            modelIs1D_b = 1, 
            loss = "categorical_crossentropy", 
            learningRate = 0.025,
            momentum = 0.001,
            nrTrainSamples = 100000, #is set (below) to length of the trainDataInterval, so of no importance
            trainDataInterval = [0,200000] , 
            dynSamplesTransformStyle_b = 1,
            outDtype = 'float32',
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000], #is not used
            customFlankSize_b = 0, 
            customFlankSize = 50,
            genSamples_b = 0, 
            genomeFileName = '',
            outputEncodedOneHot_b = 1,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
            genSamplesFromRandomGenome_b = 0, 
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            augmentWithRevComplementary_b = 0, 
            augmentTestDataWithRevComplementary_b = 0,
            inclFrqModel_b = 0,
            insertFrqModel_b = 0,
            frqModelFileName = '',
            frqSoftmaxed_b = 0,
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0,
            optimizer = 'ADAM',
            batchSize = 128, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=4,
            letterShape = 4, # size of the word
            lengthWindows = [2, 3, 4, 4, 5, 10],
            hiddenUnits= [50], #for conv1d and conv2d only the first entry is used 
            dropoutVal= 0.25,
            dropoutLastLayer_b = 0,
            nrFilters = [50, 50, 50, 50, 50, 50],    
            padding = 'same',  
            pool_b = 0,
            maxPooling_b = 0,
            poolAt = [],
            poolStrides = 1,
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'Conv type ... to be filled in!',
            on_binf_b = 1):
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


            
#1st version parameters:    
#    lengthWindows = 5, 7, 10, 15, 20, 25
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 30, 20, 10
    
#2nd version parameter:
#    lengthWindows = 3, 5, 7
#    nrHiddenUnits = 25
#    nrFilters = 25, 15, 10
      
##3rd version parameter:
#    lengthWindows = 3, 3
#    nrHiddenUnits = 25
#    nrFilters = 10, 5
     
#    lengthWindows = 3, 3, 4, 6, 8, 10, 15, 20, 30
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 45, 40, 30, 25, 20, 15
     
     
    
    #Using the getData-fct to fetch data:
#    fname=root + r"trShort.dat"
#    tname=root + r"tsShort.dat"
#    vname=root + r"vlShort.dat"
#      
#    X,Y = dnaNet.getData2(fname, letterShape, sizeOutput, outputType=float)  
#    sizeInputput = X.shape[1] 
#        
#    Xt,Yt = dnaNet.getData2(tname, letterShape, sizeOutput , outputType=float)  
#    
#    
    
    
    #repeat a training/testing round the set nr of times; after first round the model (from the previous round) is reloaded
    lTrainDataInterval = trainDataInterval[1] - trainDataInterval[0]
    nrTrainSamples = lTrainDataInterval
#    lTestDataInterval = testDataInterval[1] - testDataInterval[0]
    for n in range(firstIterNr, firstIterNr + nrOuterLoops):
        
        print("Now at outer iteration: ", n)
        
        modelFileName = rootOutput + modelName + '_bigLoopIter' + str(n)

        trainDataInterval =  [n*lTrainDataInterval, (n+1)*lTrainDataInterval]
        print("trainDataInterval ", trainDataInterval)
        testDataInterval = [(n+1)*lTrainDataInterval, (n+1)*lTrainDataInterval + nrTestSamples]
        print("testDataInterval ", testDataInterval)
          
    #    #If the model is 2d we convert all data to images: seq of consisting of two flanks
    #    #becomes a flankSize-by-flankSize matrix of 4-by-4 matrices, the latter being 
    #    #the one-hot encoding of the 16 possible letter pairs 
    #    convertToPict_b = 0
    #    if modelIs1D_b == 0: #model is 2D
    #
    #        convertToPict_b = 1
        
        
    
        if genSamples_b > 0.5: #generate a set of random samples from genome or random data acc to the input/the sizes set
    
            #if a genomeFileName is specified, use that genome:
            if len(genomeFileName) > 0:
                
                fromGenome_b = 1
#                
#                startAtPosition = trainDataInterval[0]
#                endAtPosition = trainDataInterval[1]
#                
#                #read in the genome sequence:
#                if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
#                    genomeArray, repeatInfoArray, genomeString = encodeGenome(genomeFileName, startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0)
#                    lGenome = len(genomeArray)
#                    genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
#                elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
#                    genomeDictArray, repeatInfoDictArray, genomeDictString = encodeGenome(genomeFileName,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1)
#                    if len(genomeDictArray.keys()) > 1:
#                        print "Warning: more than one chromosome has been selected"
#                    chromo = genomeDictArray.keys()[0]
#                    genomeArray = genomeDictArray[chromo]
#                    genomeString = genomeDictString[chromo]
#                    lGenome = len(genomeArray)
#                    genomeSeqSourceTrain = chromo
#                    
#                print "lGenome: %d" % lGenome            
#                
#            else:
#                print "This code pt only runs with supplied genome data; so provide a genomeFileName"
#    
#    
            batch_size = batchSize
            #print batch_size, nrEpoch
                    
            if inclFrqModel_b == 1:
                
                if insertFrqModel_b != 1:
                        
                    #We need to split the data in the part input to the conv layer and 
                    #the part which is the output of frq model; the same is done for the
                    #test data (below):
                    sizeInputConv = 2*(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel)
                    
                    Xconv = np.zeros(shape = (batchSize, sizeInputConv, letterShape))
                    Xfrq = np.zeros(shape = (batchSize, 1, letterShape))
                
                elif insertFrqModel_b == 1:
                    
                    sizeInput = 2*customFlankSize  + 1
                                                
                
            else:  
            
                sizeInput = 2*customFlankSize 
#                
#                
#    
#            #we fetch the output from the frq model if we want to include it in the training and testing; 
#            #the test set shall also include the frq model output if so; the data for testing is loaded after
#            #the training is done (below) so as to avoid spending the memory needed for the test data during 
#            #the training part: 
#            frqModelDict = {}   
#            if inclFrqModel_b == 1:
#                    
#                frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel, applySoftmax_b = frqSoftmaxed_b)          
#                         
#            
#            #Dynamically fetch small sample batches; this runs in an infinite loop
#            #in parallel to the fit_generator call below (and stops when that is done)
#            if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
#            
#                flankSizeOut = customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel
#                
#            else:
#                
#                flankSizeOut = customFlankSize
#                
#            #the dyn-sampling can be done "transformation style": this means that a block of memory dedicated to
#            #the training data and one for the labels, are "allocated" once; these blocks are then resued for every
#            #batch during the training. If not running "transforamtion style" the generator will allocate a new pair
#            #of blocks for each batch
#            if dynSamplesTransformStyle_b == 1:
#            
#                if outputEncodedOneHot_b == 1:
#                    
#                    if inclFrqModel_b == 1:
#                        
#                        if augmentWithRevComplementary_b == 1:
#                            X = np.zeros(shape = (2*batchSize, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
#                            Y = np.zeros(shape = (2*batchSize, 4), dtype = outDtype ) #to hold the labels
#                        else:
#                            X = np.zeros(shape = (batchSize, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
#                            Y = np.zeros(shape = (batchSize, 4), dtype = outDtype ) #to hold the label
#            
#                    else:
#            
#                        if augmentWithRevComplementary_b == 1:
#                            X = np.zeros(shape = (2*batchSize, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
#                            Y = np.zeros(shape = (2*batchSize, 4), dtype = outDtype ) #to hold the labels
#                        else:
#                            X = np.zeros(shape = (batchSize, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
#                            Y = np.zeros(shape = (batchSize, 4), dtype = outDtype ) #to hold the labels
#            
#                elif outputEncodedInt_b == 1:
#                    
#                    if augmentWithRevComplementary_b == 1:
#                        X = np.zeros(shape = (2*batchSize, 2*flankSizeOut), dtype = outDtype ) #to hold the flanks
#                        Y = np.zeros(shape = (2*batchSize,1), dtype = outDtype ) #to hold the labels
#                    else:
#                        X = np.zeros(shape = (batchSize, 2*flankSizeOut), dtype = outDtype ) #to hold the flanks
#                        Y = np.zeros(shape = (batchSize), dtype = outDtype ) #to hold the labels
#                
#            else: #put in dummies
#                
#                X = 0
#                Y = 0
#                            
#    #        def myGenerator(customFlankSize,batchSize, inclFrqModel_b):               
#            def myGenerator(X,Y):               
#                
#                while 1:
#                    
#                    if dynSamplesTransformStyle_b == 0:
#                        
#    #                    print "I'm using the generator transform style"
#                    
#                        X,Y = genSamplesForDynamicSampling_I(transformStyle_b = dynSamplesTransformStyle_b, nrSamples = batchSize, genomeArray = genomeArray, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
#                                                           genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
#    
#    
#                    elif dynSamplesTransformStyle_b == 1:
#                        
#                        X,Y = genSamplesForDynamicSampling_I(transformStyle_b = dynSamplesTransformStyle_b, X = X, Y = Y, nrSamples = batchSize, genomeArray = genomeArray, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
#                                                           genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
#                        
#    
#    #                sizeInput = X.shape[1]
#    
#                                        
#    #                print np.sum(Y)
#    #                print Y
#    
#                    if inclFrqModel_b == 1 and insertFrqModel_b != 1:
#                            
#                        Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
#                        Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
#                        Xfrq[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
#    
#    #                    print np.sum(Xfrq)
#    #                    print X.shape, Y.shape
#                        
#    #                    XsplitList = []            
#    #                    for i in range(batchSize):
#    #                        
#    #                        print Xfrq[i].shape, Xconv[i].shape
#    #                        
#    #                        XsplitList.append([Xfrq[i], Xconv[i]])
#    #            
#    #                    Xsplit = np.asarray(XsplitList)        
#    #                    
#    #                    raw_input("Sis du er saa n..")
#    #                    merge = Concatenate()([Xfrq,Xconv])
#    #                    raw_input("Sis du er saa ..")
#    #                    Xsplit = Reshape((-1,))(merge)
#    #                    
#    #                    print(Xsplit._keras_shape)
#    #                    raw_input("Sis du er saa ..")
#    
#    #                    print Xfrq.shape, Xconv.shape
#    #                    print np.asarray([Xfrq, Xconv]).shape
#    #                    yield(Xsplit, Y)
#                        yield([Xfrq, Xconv],Y)
#                        
#                    
#                    else:
#                        
#                        yield(X, Y)
#        #            print "X shape", X.shape
            
    
    
        #reload the model from iter            
        modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n)
        net = model_from_json(open(modelFileNamePrevious).read())
        net.load_weights(modelFileNamePrevious +'.h5')

        print("I've now reloaded the model  iteration: ", n)

            
        if optimizer == 'SGD':
            
            print("I'm using the SGD optimizer")
            optUsed = SGD(lr= learningRate, decay=1e-6, momentum= momentum, nesterov=True)
            #sgd = SGD(lr=0.01)
    
        elif optimizer =='ADAM':
        
            print("I'm using the ADAM optimizer")
            optUsed = Adam(lr= learningRate)
        
    
        elif optimizer =='RMSprop':
        
            print("I'm using the RMSprop optimizer with default rho of 0.9 and decay of 0")
            optUsed = RMSprop(lr= learningRate)
            
        
        
        net.compile(loss=loss, optimizer=optUsed, metrics=['accuracy'])
        
        #net.compile(loss='categorical_crossentropy', optimizer='Adam')
        #net.compile(loss='binary_crossentropy', optimizer=sgd)
        #xor.compile(loss="hinge", optimizer=sgd)
        #xor.compile(loss="binary_crossentropy", optimizer=sgd)
          
        #net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, show_accuracy=True, verbose=0)
    #    history = net.fit_generator(X, Y, batch_size=batch_size, epochs = nrEpochs, verbose=1, validation_data=(Xv,Yv) )
    
        
#        history=net.fit_generator(myGenerator(X= X,Y= Y), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
#        
#    
#        # list all data in history
#        print(history.history.keys())
#        # summarize history for accuracy
#        plt.figure()
#        plt.plot(history.history['acc'])
#        plt.title('model accuracy')
#        plt.ylabel('accuracy')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        plt.savefig(modelFileName + '_training_validation_acc_vs_epoch' + '.pdf')
#    #    plt.show()
#        plt.close()
#        # summarize history for loss
#        plt.figure()
#        plt.plot(history.history['loss'])
#        plt.title('model loss')
#        plt.ylabel('loss')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        plt.savefig(modelFileName + '_training_validation_loss_vs_epoch' + '.pdf')
#    #    plt.show()
#        plt.close()
#    
#        if save_model_b == 1:
#             
#            json_string = net.to_json()
#            open(modelFileName, 'w').write(json_string)
#            net.save_weights(modelFileName + '.h5',overwrite=True)
    
        #test it. First read in the test data:
        frqModelDict = {}   
        genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
        if inclFrqModel_b == 1:
    
            #Read in the test data we avoid the chromos used for training:    
            avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
            Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
                                                    flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
    
            if insertFrqModel_b != 1:
                
                #Split the test data as the training data:
                nrOfTestSamples = Xt.shape[0]
                Xconv_t = np.zeros(shape = (nrOfTestSamples, sizeInputConv, letterShape))
                Xfrq_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
        
                Xconv_t[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                Xconv_t[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                Xfrq_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
    
    #            XsplitList_t = []            
    #            for i in range(nrOfTestSamples):
    #                
    #                XsplitList_t.append([Xfrq_t[i], Xconv_t[i]])
    #            
    #            Xsplit_t = np.asarray(XsplitList_t)
                
                
    #            print Xconv_t.shape, Xfrq_t.shape            
            
        else:
    
            #Read in the test data we avoid the chromos used for training:    
            avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
            Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
            if Xt.shape[0] < nrTestSamples:
                m = 1
                while Xt.shape[0] < nrTestSamples:
                    
                    testDataInterval = [(n+1 +m)*lTrainDataInterval, (n+1+m)*lTrainDataInterval + nrTestSamples]
                    print("Too few samples in testDataInterval so consider new interval: ", testDataInterval)
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
       
        #when if augmentWithRevComplementary_b == 1 the generated batches contain 2*batchSize samples:
        if augmentWithRevComplementary_b == 0:
            batchSizeReal = batchSize
        else:
            batchSizeReal = 2*batchSize
    
    
        if inclFrqModel_b == 1:
            
            if insertFrqModel_b == 1:
                
                score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
            
            else:
                
                score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
        else:
            
            score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
    
        print('Test score:', score)
        print('Test accuracy:', acc)
        
    
                         
            
        #Write run-data to txt-file for documentation of the run:
        runDataFileName = modelFileName + '_runData' + '.txt'
        runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
        
        s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
        s += modelDescription  + "\n"   
        s += 'Performance on test set, loss and accuracy resp.: ' + str(score) + ' ' + str(acc) + "\n"   
        if save_model_b == 1:
            s+= 'Model data obtained after training the model are recorded in: ' +  rootOutput + modelName + ' and ' + rootOutput + modelName + '.h5\n' 
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #Which genome data were used?:
        if genSamples_b > 0.5:
            s = "Samples generated with python code from real genome." + "\n"
            s += "Genome data in file: " + genomeFileName + "\n"
            s += 'inclFrqModel_b: ' + str(inclFrqModel_b) + "\n"
            s += 'frqModelFileName: ' + frqModelFileName + "\n"  
            s += "Letters are one-hot encoded" + "\n"
            s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
            s += "trainDataInterval:" + str(trainDataInterval)  + "\n"    
            s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
            s += "testDataInterval:" + str(testDataInterval)  + "\n" 
            if onlyOneRandomChromo_b == 1:
                s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
                s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
                s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
            else:
                s += "Read in the whole genome sequence" + "\n"
            s += "shuffle_b = " + str(shuffle_b) + "\n"
            s += "inner_b = " + str(inner_b) + "\n"
            s += "shuffleLength = " + str(shuffleLength) +  "\n"
            s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
            s += "trainDataInterval:" + str(trainDataInterval)  + "\n"  
            s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
            s += "testDataInterval:" + str(testDataInterval)  + "\n" 
         
        elif genSamplesFromRandomGenome_b > 0.5:
            
            s = "Samples from random genome, all generated with python code." + "\n"
            s += "Genome data in file: " + randomGenomeFileName
                
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #various params:    
        s= 'loss = "categorical_crossentropy"\n'       
        s += 'trainDataInterval: ' + str(trainDataInterval) + "\n"
        s += 'testDataInterval: ' + str(testDataInterval) + "\n" 
        s += 'customFlankSize_b: ' + str(customFlankSize_b) + "\n" 
        s += 'customFlankSize: ' + str(customFlankSize) + "\n" 
        s += 'genSamples_b: ' + str(genSamples_b) + "\n" 
        s += 'genomeFileName: ' + genomeFileName + "\n" 
        s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
        s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
        s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n" 
        s += "avoidChromo: " + str(avoidChromo)  + "\n" 
        s += 'genSamplesFromRandomGenome_b: ' + str(genSamplesFromRandomGenome_b) + "\n" 
        s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
        s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
        s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
        s += 'Optimizer: ' + optimizer + "\n"
        s += 'learning rate: ' + str(learningRate)  + "\n" 
        s += 'momentum: ' + str(momentum) + "\n" 
        s += 'batchSize: ' + str(batchSize) + "\n"
        s += 'dropoutVal: ' + str(dropoutVal) + "\n"
        s += 'dropoutLastLayer_b: ' + str(dropoutLastLayer_b) + "\n"
        s += 'padding: ' + padding + "\n"
        s += 'pool_b: ' +  str(pool_b) + "\n"
        s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
        s += 'poolAt: ' +  str(poolAt) + "\n"
        s += "dynSamplesTransformStyle_b:" + str(dynSamplesTransformStyle_b) + "\n"
        s += 'stepsPerEpoch: ' + str(stepsPerEpoch) + "\n" 
        s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
        s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
        s += 'letterShape: ' + str(letterShape) + "\n" 
        s += 'save_model_b: ' + str(save_model_b) + "\n" 
        s += 'modelName: ' + modelName + "\n" 
        s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
        
        runDataFile.write(s)
            
        #Params for net:
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        s = 'lengthWindows: ' + str(lengthWindows)  + "\n" 
        s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
        s += 'nrFilters: ' + str(nrFilters)
        
        runDataFile.write(s)
        
        runDataFile.close()
      
        
        
def allInOneWithDynSampling_ConvModel_I(nrOuterLoops = 1,
                                        firstIterNr = 0,
                                        nrOfRepeats = 1,
                                        firstRepeatNr = 0,
            loadModelFromFile_b = 0,
            modelFileName = '',
            modelIs1D_b = 1, 
            loss = "categorical_crossentropy", 
            learningRate = 0.025,
            momentum = 0.001,
            trainDataIntervalStepSize = 100000, 
            trainDataInterval0 = [0,200000] , 
            dynSamplesTransformStyle_b = 1,
            outDtype = 'float32',
            testDataIntervalIdTotrainDataInterval_b = 0,
            nrTestSamples = 20000,
            testDataInterval0 = [400000, 600000], #is not used
            customFlankSize_b = 0, 
            customFlankSize = 50,
            genSamples_b = 0, 
            genomeFileName = '',
            exonicInfoBinaryFileName  = '' ,
            outputEncodedOneHot_b = 1,
            labelsCodetype = 0,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
            genSamplesFromRandomGenome_b = 0, 
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            augmentWithRevComplementary_b = 0, 
            augmentTestDataWithRevComplementary_b = 0,
            inclFrqModel_b = 0,
            insertFrqModel_b = 0,
            frqModelFileName = '',
            frqSoftmaxed_b = 0,
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0,
            optimizer = 'ADAM',
            batchSize = 128, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=4,
            letterShape = 4, # size of the word
            lengthWindows = [2, 3, 4, 4, 5, 10],
            hiddenUnits= [50], #for conv1d and conv2d only the first entry is used 
            dropoutVal= 0.25,
            dropoutLastLayer_b = 0,
            nrFilters = [50, 50, 50, 50, 50, 50],    
            padding = 'same',  
            pool_b = 0,
            maxPooling_b = 0,
            poolAt = [],
            poolStrides = 1,
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'Conv type ... to be filled in!',
            on_binf_b = 1):
    
    '''
        labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding and not including the frq model 
                (inclFrqModel_b = 0).
                
    '''
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


            
#1st version parameters:    
#    lengthWindows = 5, 7, 10, 15, 20, 25
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 30, 20, 10
    
#2nd version parameter:
#    lengthWindows = 3, 5, 7
#    nrHiddenUnits = 25
#    nrFilters = 25, 15, 10
      
##3rd version parameter:
#    lengthWindows = 3, 3
#    nrHiddenUnits = 25
#    nrFilters = 10, 5
     
#    lengthWindows = 3, 3, 4, 6, 8, 10, 15, 20, 30
#    nrHiddenUnits = 25
#    nrFilters = 50, 45, 40, 45, 40, 30, 25, 20, 15
     
     
    
    #Using the getData-fct to fetch data:
#    fname=root + r"trShort.dat"
#    tname=root + r"tsShort.dat"
#    vname=root + r"vlShort.dat"
#      
#    X,Y = dnaNet.getData2(fname, letterShape, sizeOutput, outputType=float)  
#    sizeInputput = X.shape[1] 
#        
#    Xt,Yt = dnaNet.getData2(tname, letterShape, sizeOutput , outputType=float)  
#    
#    
    
    
    #repeat a training/testing round the set nr of times; after first round the model (from the previous round) is reloaded
#    lTrainDataInterval = trainDataInterval[1] - trainDataInterval[0]
#    lTestDataInterval = testDataInterval[1] - testDataInterval[0]  
    historyTotal = {} #for recording training performance (acc/loss) across all iterations/repeats
    historyTotal['acc'] = []
    historyTotal['loss'] = []  
    testHistoryTotal = {} #for recording testing performance (acc/loss) across all iterations/repeats
    testHistoryTotal['acc'] = []
    testHistoryTotal['loss'] = []      
    for n in range(firstIterNr, nrOuterLoops):
        
        print("Now at outer iteration: ", n)
        
        modelFileName = rootOutput + modelName + '_bigLoopIter' + str(n)

        trainDataInterval =  [trainDataInterval0[0] + n*trainDataIntervalStepSize, trainDataInterval0[1] + n*trainDataIntervalStepSize]
        if testDataIntervalIdTotrainDataInterval_b == 1:
            
            testDataInterval = trainDataInterval
        
        else:
        
            testDataInterval = [trainDataInterval[1], trainDataInterval[1] + nrTestSamples]

            
        print("trainDataInterval ", trainDataInterval)
        print("testDataInterval ", testDataInterval)
          
    #    #If the model is 2d we convert all data to images: seq of consisting of two flanks
    #    #becomes a flankSize-by-flankSize matrix of 4-by-4 matrices, the latter being 
    #    #the one-hot encoding of the 16 possible letter pairs 
    #    convertToPict_b = 0
    #    if modelIs1D_b == 0: #model is 2D
    #
    #        convertToPict_b = 1
        
    
        if genSamples_b > 0.5: #generate a set of random samples from genome or random data acc to the input/the sizes set
    
            #if a genomeFileName is specified, use that genome:
            if len(genomeFileName) > 0:
                
                fromGenome_b = 1
                
                startAtPosition = trainDataInterval[0]
                endAtPosition = trainDataInterval[1]
                
                #read in the genome sequence:
                if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
                    genomeArray, repeatArray, exonicArray, genomeString = encodeGenome(fileName = genomeFileName, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0)
                    lGenome = len(genomeArray)
                    genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
                elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                    genomeDictArray, repeatInfoDictArray, exonicInfoDictArray, genomeDictString = encodeGenome(fileName = genomeFileName, exonicInfoBinaryFileName  = exonicInfoBinaryFileName ,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1)
                    if len(genomeDictArray.keys()) > 1:
                        print("Warning: more than one chromosome has been selected")
                    chromo = genomeDictArray.keys()[0]
                    genomeArray = genomeDictArray[chromo]
                    repeatArray = repeatInfoDictArray[chromo]
                    exonicArray = exonicInfoDictArray[chromo]
                    genomeString = genomeDictString[chromo]
                    lGenome = len(genomeArray)
                    genomeSeqSourceTrain = chromo
                    
                print("lGenome: %d" % lGenome)
                
            else:
                print("This code pt only runs with supplied genome data; so provide a genomeFileName")

   
            batch_size = batchSize
            #print batch_size, nrEpoch
                    
            if inclFrqModel_b == 1:
                
                if insertFrqModel_b != 1:
                        
                    #We need to split the data in the part input to the conv layer and 
                    #the part which is the output of frq model; the same is done for the
                    #test data (below):
                    sizeInputConv = 2*(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel)
                    
                    Xconv = np.zeros(shape = (batchSize, sizeInputConv, letterShape))
                    Xfrq = np.zeros(shape = (batchSize, 1, letterShape))
                
                elif insertFrqModel_b == 1:
                    
                    sizeInput = 2*customFlankSize  + 1
                                                
                
            else:  
            
                sizeInput = 2*customFlankSize 
                
                
            #output size depends on what we want to predict (base or base pair or pyri/puri)
            if labelsCodetype == 0:
                sizeOutput = 4
            elif labelsCodetype == 1 or labelsCodetype == -1:
                sizeOutput = 2
            elif labelsCodetype == 2:
                sizeOutput = 3
            
            #we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so; the data for testing is loaded after
            #the training is done (below) so as to avoid spending the memory needed for the test data during 
            #the training part: 
            frqModelDict = {}   
            if inclFrqModel_b == 1:
                    
                frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel, applySoftmax_b = frqSoftmaxed_b)          
                         
            
            #Dynamically fetch small sample batches; this runs in an infinite loop
            #in parallel to the fit_generator call below (and stops when that is done)
            if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
            
                flankSizeOut = customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel
                
            else:
                
                flankSizeOut = customFlankSize
                
            #the dyn-sampling can be done "transformation style": this means that a block of memory dedicated to
            #the training data and one for the labels, are "allocated" once; these blocks are then resued for every
            #batch during the training. If not running "transforamtion style" the generator will allocate a new pair
            #of blocks for each batch
            if dynSamplesTransformStyle_b == 1:
            
                if outputEncodedOneHot_b == 1:
                    
                    if inclFrqModel_b == 1:
                        
                        if augmentWithRevComplementary_b == 1:
                            X = np.zeros(shape = (2*batchSize, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
                            Y = np.zeros(shape = (2*batchSize, sizeOutput), dtype = outDtype ) #to hold the labels
                        else:
                            X = np.zeros(shape = (batchSize, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
                            Y = np.zeros(shape = (batchSize, sizeOutput), dtype = outDtype ) #to hold the labels
            
                    else:
            
                        if augmentWithRevComplementary_b == 1:
                            X = np.zeros(shape = (2*batchSize, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                            Y = np.zeros(shape = (2*batchSize, sizeOutput), dtype = outDtype ) #to hold the labels
                        else:
                            X = np.zeros(shape = (batchSize, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                            Y = np.zeros(shape = (batchSize, sizeOutput), dtype = outDtype ) #to hold the labels
            
                elif outputEncodedInt_b == 1:
                    
                    if augmentWithRevComplementary_b == 1:
                        X = np.zeros(shape = (2*batchSize, 2*flankSizeOut), dtype = outDtype ) #to hold the flanks
                        Y = np.zeros(shape = (2*batchSize,1), dtype = outDtype ) #to hold the labels
                    else:
                        X = np.zeros(shape = (batchSize, 2*flankSizeOut), dtype = outDtype ) #to hold the flanks
                        Y = np.zeros(shape = (batchSize), dtype = outDtype ) #to hold the labels
                
            else: #put in dummies
                
                X = 0
                Y = 0
                            
    #        def myGenerator(customFlankSize,batchSize, inclFrqModel_b):               
            def myGenerator(X,Y):               
                
                while 1:
                    
                    if dynSamplesTransformStyle_b == 0:
                        
    #                    print "I'm using the generator transform style"
                    
                        X,Y = genSamplesForDynamicSampling_I(transformStyle_b = dynSamplesTransformStyle_b, nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
                                                           genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
    
    
                    elif dynSamplesTransformStyle_b == 1:
                        
                        X,Y = genSamplesForDynamicSampling_I(transformStyle_b = dynSamplesTransformStyle_b, X = X, Y = Y, nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
                                                           genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                        
    
    #                sizeInput = X.shape[1]
    
                                        
    #                print np.sum(Y)
    #                print Y
    
                    if inclFrqModel_b == 1 and insertFrqModel_b != 1:
                            
                        Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                        Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                        Xfrq[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
    
    #                    print np.sum(Xfrq)
    #                    print X.shape, Y.shape
                        
    #                    XsplitList = []            
    #                    for i in range(batchSize):
    #                        
    #                        print Xfrq[i].shape, Xconv[i].shape
    #                        
    #                        XsplitList.append([Xfrq[i], Xconv[i]])
    #            
    #                    Xsplit = np.asarray(XsplitList)        
    #                    
    #                    raw_input("Sis du er saa n..")
    #                    merge = Concatenate()([Xfrq,Xconv])
    #                    raw_input("Sis du er saa ..")
    #                    Xsplit = Reshape((-1,))(merge)
    #                    
    #                    print(Xsplit._keras_shape)
    #                    raw_input("Sis du er saa ..")
    
    #                    print Xfrq.shape, Xconv.shape
    #                    print np.asarray([Xfrq, Xconv]).shape
    #                    yield(Xsplit, Y)
                        yield([Xfrq, Xconv],Y)
                        
                    
                    else:
                        
                        yield(X, Y)
        #            print "X shape", X.shape
            
        
        #output size depends on what we want to predict (base or base pair or pyri/puri)
        if labelsCodetype == 0:
            sizeOutput = 4
        elif labelsCodetype == 1 or labelsCodetype == -1:
            sizeOutput = 2
        elif labelsCodetype == 2:
            sizeOutput = 3
        
        for k in range(firstRepeatNr, nrOfRepeats):
            
            print("Now at outer iteration %d ,repeat %d" % (n,k))
    
            #in first outer-iteration build the model; thereafter reload the latest stored version (saved below)
            if n == 0 and k == 0:
                loadModelFromFile_b = 0
            else:
                loadModelFromFile_b = 1
            
            if inclFrqModel_b == 1:
        
                
                if insertFrqModel_b == 1:
                    
                    net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b)
        
                else: #Merge the frq model output into (one of) the last layers 
        
                    net = makeConv1DmodelMergedWithFrqModel(frqModelOutputSize = 1, sequenceLength = sizeInputConv, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt)       
        
                if loadModelFromFile_b == 1: 
        
    
                    #reload the model from previous iter/repeat
                    if k == 0:
                        modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n-1) + '_repeatNr' + str(nrOfRepeats-1)
                    else:
                        modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n) + '_repeatNr' + str(k-1)
                        
                    net = model_from_json(open(modelFileNamePrevious).read())
                    net.load_weights(modelFileNamePrevious +'.h5')
            
                    print("I've now reloaded the model from the previous iteration: ", modelFileNamePrevious)
    
                
            else:  
                      
                if loadModelFromFile_b == 0: 
                    
                    if modelIs1D_b == 1: 
                        
                        net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt)
                
                    else:
                        
                        net = makeConv2Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, padding = padding, pool_b = pool_b, poolStrides = (poolStrides, poolStrides), maxPooling_b = maxPooling_b, poolAt = poolAt)
        
                else:
    
                    if k == 0:
                        modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n-1) + '_repeatNr' + str(nrOfRepeats-1)
                    else:
                        modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n) + '_repeatNr' + str(k-1)
                        
                    net = model_from_json(open(modelFileNamePrevious).read())
                    net.load_weights(modelFileNamePrevious +'.h5')
            
                    print("I've now reloaded the model from the previous iteration: ", modelFileNamePrevious)
                
                
            if optimizer == 'SGD':
                
                print("I'm using the SGD optimizer")
                optUsed = SGD(lr= learningRate, decay=1e-6, momentum= momentum, nesterov=True)
                #sgd = SGD(lr=0.01)
        
            elif optimizer =='ADAM':
            
                print("I'm using the ADAM optimizer")
                optUsed = Adam(lr= learningRate)
            
        
            elif optimizer =='RMSprop':
            
                print("I'm using the RMSprop optimizer with default rho of 0.9 and decay of 0")
                optUsed = RMSprop(lr= learningRate)
                
            
            
            net.compile(loss=loss, optimizer=optUsed, metrics=['accuracy'])
            
            #net.compile(loss='categorical_crossentropy', optimizer='Adam')
            #net.compile(loss='binary_crossentropy', optimizer=sgd)
            #xor.compile(loss="hinge", optimizer=sgd)
            #xor.compile(loss="binary_crossentropy", optimizer=sgd)
              
            #net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, show_accuracy=True, verbose=0)
        #    history = net.fit_generator(X, Y, batch_size=batch_size, epochs = nrEpochs, verbose=1, validation_data=(Xv,Yv) )
        
            
            history=net.fit_generator(myGenerator(X= X,Y= Y), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
            
            if save_model_b == 1:
                 
                json_string = net.to_json()
                open(modelFileName + '_repeatNr' + str(k), 'w').write(json_string)
                net.save_weights(modelFileName + '_repeatNr' + str(k) + '.h5',overwrite=True)

        
            # list all data in history
            print(history.history.keys())

            #dump the info:
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_acc_vs_epoch.p'
            pickle.dump( history.history['acc'], open(dumpFile, "wb") )
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_loss_vs_epoch.p'
            pickle.dump( history.history['loss'], open(dumpFile, "wb") )
            

            # summarize history for accuracy
            plt.figure()
            plt.plot(history.history['acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_training_acc_vs_epoch' + '.pdf')
        #    plt.show()
            plt.close()
            # summarize history for loss
            plt.figure()
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_training_loss_vs_epoch' + '.pdf')
        #    plt.show()
            plt.close()


            #record and plot the total performance, ie up to this iter/repeat:
            historyTotal['acc'].extend(history.history['acc'])
            historyTotal['loss'].extend(history.history['loss'])

            #dump the info:
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_acc_vs_allEpochs.p' 
            pickle.dump( historyTotal['acc'], open(dumpFile, "wb") )
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_loss_vs_allEpochs.p' 
            pickle.dump( historyTotal['loss'], open(dumpFile, "wb") )
            #.. and plot as above:
             # summarize history for accuracy
            plt.figure()
            plt.plot(historyTotal['acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_training_acc_vs_allEpochs' + '.pdf')
        #    plt.show()
            plt.close()
            # summarize history for loss
            plt.figure()
            plt.plot(historyTotal['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_training_loss_vs_allEpochs' + '.pdf')
        #    plt.show()
            plt.close()
        
            #test it. 
            #If so desired, we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so: 
            if testDataIntervalIdTotrainDataInterval_b == 0:
                
                #First read in the test data:
                if inclFrqModel_b == 1:
        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
                                                            flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
            
                    if insertFrqModel_b != 1:
                        
                        #Split the test data as the training data:
                        nrOfTestSamples = Xt.shape[0]
                        Xconv_t = np.zeros(shape = (nrOfTestSamples, sizeInputConv, letterShape))
                        Xfrq_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
                
                        Xconv_t[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                        Xconv_t[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                        Xfrq_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
            
            #            XsplitList_t = []            
            #            for i in range(nrOfTestSamples):
            #                
            #                XsplitList_t.append([Xfrq_t[i], Xconv_t[i]])
            #            
            #            Xsplit_t = np.asarray(XsplitList_t)
                        
                        
            #            print Xconv_t.shape, Xfrq_t.shape            
                    
                else:
            
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
               
                          
        
        
                #when if augmentWithRevComplementary_b == 1 the generated batches contain 2*batchSize samples:
                if augmentWithRevComplementary_b == 0:
                    batchSizeReal = batchSize
                else:
                    batchSizeReal = 2*batchSize
            
            
                if inclFrqModel_b == 1:
                    
                    if insertFrqModel_b == 1:
                        
                        score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
                    
                    else:
                        
                        score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
                else:
                    
                    score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
                        
                    
            elif testDataIntervalIdTotrainDataInterval_b == 1: #we test using the dynamic sampling
                      
                #dummy valeus:
                Xt = 0
                Yt = 0
                
                score, acc = net.evaluate_generator(myGenerator(Xt,Yt), steps = np.int(float(nrTestSamples)/batchSize))
   


        
    #        if inclFrqModel_b == 1:
    #            
    #            if insertFrqModel_b == 1:
    #                
    #                score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
    #            
    #            else:
    #                
    #                score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batch_size, verbose=1)
    #        else:
    #            
    #            score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
    
#            #when if augmentWithRevComplementary_b == 1 the generated batches contain 2*batchSize samples:
#            if augmentWithRevComplementary_b == 0:
#                batchSizeReal = batchSize
#            else:
#                batchSizeReal = 2*batchSize
#        
#        
#            if inclFrqModel_b == 1:
#                
#                if insertFrqModel_b == 1:
#                    
#                    score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
#                
#                else:
#                    
#                    score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
#            else:
#                
#                score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)

    
        
            print('Test score:', score)
            print('Test accuracy:', acc)
            
            #record and plot the total test performance, ie up to this iter/repeat:
            testHistoryTotal['acc'].append(acc)
            testHistoryTotal['loss'].append(score)
            #dumpt the info
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_testing_acc_vs_allEpochs.p' 
            pickle.dump( testHistoryTotal['acc'], open(dumpFile, "wb") )
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_testing_loss_vs_allEpochs.p' 
            pickle.dump( testHistoryTotal['loss'], open(dumpFile, "wb") )
            #.. and plot as above:
             # summarize history for accuracy
            plt.figure()
            plt.plot(testHistoryTotal['acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['test'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_testing_acc_vs_allEpochs' + '.pdf')
        #    plt.show()
            plt.close()
            # summarize history for loss
            plt.figure()
            plt.plot(testHistoryTotal['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['test'], loc='upper left')
            plt.savefig(modelFileName + '_repeatNr' + str(k) + '_testing_loss_vs_allEpochs' + '.pdf')
        #    plt.show()
            plt.close()
        
                         
            
        #Write run-data to txt-file for documentation of the run (outer iter):
        runDataFileName = modelFileName + '_runData' + '.txt'
        runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
        
        s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
        s += modelDescription  + "\n"   
        s += 'Performance on test set, loss and accuracy resp.: ' + str(score) + ' ' + str(acc) + "\n"   
        if save_model_b == 1:
            s+= 'Model data obtained after training the model are recorded in: ' +  rootOutput + modelName + ' and ' + rootOutput + modelName + '.h5\n' 
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #Which genome data were used?:
        if genSamples_b > 0.5:
            s = "Samples generated with python code from real genome." + "\n"
            s += "Genome data in file: " + genomeFileName + "\n"
            s += 'inclFrqModel_b: ' + str(inclFrqModel_b) + "\n"
            s += 'frqModelFileName: ' + frqModelFileName + "\n"  
            s += "Letters are one-hot encoded" + "\n"
            s += "Labels are encoded as type" + str(labelsCodetype) + "\n"
            if onlyOneRandomChromo_b == 1:
                s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
                s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
                s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
            else:
                s += "Read in the whole genome sequence" + "\n"
            s += "shuffle_b = " + str(shuffle_b) + "\n"
            s += "inner_b = " + str(inner_b) + "\n"
            s += "shuffleLength = " + str(shuffleLength) +  "\n"
            s += "trainDataIntervalStepSize:" + str(trainDataIntervalStepSize)  + "\n"
            s += "trainDataInterval:" + str(trainDataInterval)  + "\n"  
            s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
            s += "testDataInterval:" + str(testDataInterval)  + "\n" 
         
        elif genSamplesFromRandomGenome_b > 0.5:
            
            s = "Samples from random genome, all generated with python code." + "\n"
            s += "Genome data in file: " + randomGenomeFileName
                
        runDataFile.write(s)
        
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        #various params:    
        s= 'loss = "categorical_crossentropy"\n'       
        s += 'trainDataInterval: ' + str(trainDataInterval) + "\n"
        s += 'testDataInterval: ' + str(testDataInterval) + "\n" 
        s += 'customFlankSize_b: ' + str(customFlankSize_b) + "\n" 
        s += 'customFlankSize: ' + str(customFlankSize) + "\n" 
        s += 'genSamples_b: ' + str(genSamples_b) + "\n" 
        s += 'genomeFileName: ' + genomeFileName + "\n" 
        s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
        s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
        s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n" 
        s += "avoidChromo: " + str(avoidChromo)  + "\n" 
        s += 'genSamplesFromRandomGenome_b: ' + str(genSamplesFromRandomGenome_b) + "\n" 
        s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
        s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
        s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
        s += 'Optimizer: ' + optimizer + "\n"
        s += 'learning rate: ' + str(learningRate)  + "\n" 
        s += 'momentum: ' + str(momentum) + "\n" 
        s += 'batchSize: ' + str(batchSize) + "\n"
        s += 'dropoutVal: ' + str(dropoutVal) + "\n"
        s += 'dropoutLastLayer_b: ' + str(dropoutLastLayer_b) + "\n"
        s += 'padding: ' + padding + "\n"
        s += 'pool_b: ' +  str(pool_b) + "\n"
        s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
        s += 'poolAt: ' +  str(poolAt) + "\n"
        s += "dynSamplesTransformStyle_b:" + str(dynSamplesTransformStyle_b) + "\n"
        s += 'stepsPerEpoch: ' + str(stepsPerEpoch) + "\n" 
        s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
        s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
        s += 'letterShape: ' + str(letterShape) + "\n" 
        s += 'save_model_b: ' + str(save_model_b) + "\n" 
        s += 'modelName: ' + modelName + "\n" 
        s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
        
        runDataFile.write(s)
            
        #Params for net:
        s = '' #reset
        runDataFile.write(s + "\n") #insert blank line
        s = 'lengthWindows: ' + str(lengthWindows)  + "\n" 
        s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
        s += 'nrFilters: ' + str(nrFilters)
        
        runDataFile.write(s)
        
        runDataFile.close()
      

#NOT IN WORKING ORDER
def allInOneWithDynSampling_ConvModel_II(
            loadModelFromFile_b = 0,
            modelFileName = '',
            modelIs1D_b = 1, 
            loss = "categorical_crossentropy", 
            learningRate = 0.025,
            momentum = 0.001,
            nrTrainSamples = 100000,
            trainDataInterval = [0,200000] , 
            dynSamplesTransformStyle_b = 1,
            outDtype = 'float32',
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000], 
            customFlankSize_b = 0, 
            customFlankSize = 50,
            genSamples_b = 0, 
            genomeFileName = '',
#            outputEncodedOneHot_b = 1,
#            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
            genSamplesFromRandomGenome_b = 0, 
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            augmentWithRevComplementary_b = 0, 
            augmentTestDataWithRevComplementary_b = 0,
#            inclFrqModel_b = 0,
            frqModelFileName = '',
            frqSoftmaxed_b = 0,
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0,
            optimizer = 'ADAM',
            batchSize = 128, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=2,
            letterShape = 4, # size of the word
            lengthWindows = [2, 3, 4, 4, 5, 10],
            hiddenUnits= [50], #for conv1d and conv2d only the first entry is used 
            dropoutVal= 0.25,
            dropoutLastLayer_b = 0,
            nrFilters = [50, 50, 50, 50, 50, 50],    
            padding = 'same',  
            pool_b = 0,
            maxPooling_b = 0,
            poolAt = [],
            poolStrides = 1,
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'Conv type ... to be filled in!',
            on_binf_b = 1):
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"

    

    if genSamples_b > 0.5: #generate a set of random samples from genome or random data acc to the input/the sizes set

        #if a genomeFileName is specified, use that genome:
        if len(genomeFileName) > 0:
            
            fromGenome_b = 1
            
            startAtPosition = trainDataInterval[0]
            endAtPosition = trainDataInterval[1]
            
            #read in the genome sequence:
            if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
                genomeArray, repeatInfoArray, genomeString = encodeGenome(genomeFileName, startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = 1, outputEncodedInt_b = 0, outputAsDict_b = 0)
                lGenome = len(genomeArray)
                genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
            elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                genomeDictArray, repeatInfoDictArray, genomeDictString = encodeGenome(genomeFileName,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = 1, outputEncodedInt_b = 0, outputAsDict_b = 1)
                if len(genomeDictArray.keys()) > 1:
                    print("Warning: more than one chromosome has been selected")
                chromo = genomeDictArray.keys()[0]
                genomeArray = genomeDictArray[chromo]
                genomeString = genomeDictString[chromo]
                lGenome = len(genomeArray)
                genomeSeqSourceTrain = chromo
                
            print("lGenome: %d" % lGenome)
            
        else:
            print("This code pt only runs with supplied genome data; so provide a genomeFileName")


        batch_size = batchSize
        #print batch_size, nrEpoch
                
             
        #We need to split the data in the part input to the conv layer and 
        #the part which is the output of frq model; the same is done for the
        #test data (below):
        sizeInput = 2*(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel)
                                                                 

        #we fetch the output from the frq model if we want to include it in the training and testing; 
        #the test set shall also include the frq model output if so; the data for testing is loaded after
        #the training is done (below) so as to avoid spending the memory needed for the test data during 
        #the training part: 
        frqModelDict = {}   
        frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel, applySoftmax_b = frqSoftmaxed_b)          
                     
        
        #Dynamically fetch small sample batches; this runs in an infinite loop
        #in parallel to the fit_generator call below (and stops when that is done)
        flankSizeOut = customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel
            
            
        #the dyn-sampling can be done "transformation style": this means that a block of memory dedicated to
        #the training data and one for the labels, are "allocated" once; these blocks are then resued for every
        #batch during the training. If not running "transforamtion style" the generator will allocate a new pair
        #of blocks for each batch
        if dynSamplesTransformStyle_b == 1:
        
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*batchSize, 2*flankSizeOut ,4), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (2*batchSize, sizeOutput), dtype = outDtype ) #to hold the labels
            else:
                X = np.zeros(shape = (batchSize, 2*flankSizeOut ,4), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (batchSize, sizeOutput), dtype = outDtype ) #to hold the label
    
     
        else: #put in dummies
            
            X = 0
            Y = 0
                        
#        def myGenerator(customFlankSize,batchSize, inclFrqModel_b):               
        def myGenerator(X,Y):               
            
            while 1:
                
                if dynSamplesTransformStyle_b == 0:
                    
#                    print "I'm using the generator transform style"
                
                    X,Y = genSamplesForDynamicSampling_II(transformStyle_b = dynSamplesTransformStyle_b, nrSamples = batchSize, genomeArray = genomeArray, flankSize = customFlankSize, 
                                                       genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, labelSize = sizeOutput, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)


                elif dynSamplesTransformStyle_b == 1:
                    
                    X,Y = genSamplesForDynamicSampling_II(transformStyle_b = dynSamplesTransformStyle_b, X = X, Y = Y, nrSamples = batchSize, genomeArray = genomeArray, flankSize = customFlankSize, 
                                                       genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, labelSize = sizeOutput, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                    

                                    
#                print np.sum(Y)
#                print Y

#                print "X, Y shape", X.shape, Y.shape
                yield(X, Y)
#                print "X, Y shape", X.shape, Y.shapes
        

     
              
    if loadModelFromFile_b == 0: 
        
        if modelIs1D_b == 1: 
            
            net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt)
    
        else:
            
            net = makeConv2Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, padding = padding, pool_b = pool_b, poolStrides = (poolStrides, poolStrides), maxPooling_b = maxPooling_b, poolAt = poolAt)

    else:
        
        #reload the model from previous iter            
        net = model_from_json(open(modelFileName).read())
        net.load_weights(modelFileName +'.h5')

        print("I've now reloaded the model")
        
        
    if optimizer == 'SGD':
        
        print("I'm using the SGD optimizer")
        optUsed = SGD(lr= learningRate, decay=1e-6, momentum= momentum, nesterov=True)
        #sgd = SGD(lr=0.01)

    elif optimizer =='ADAM':
    
        print("I'm using the ADAM optimizer")
        optUsed = Adam(lr= learningRate)
    

    elif optimizer =='RMSprop':
    
        print("I'm using the RMSprop optimizer with default rho of 0.9 and decay of 0")
        optUsed = RMSprop(lr= learningRate)
        
    
    
    net.compile(loss=loss, optimizer=optUsed, metrics=['accuracy'])
    
    #net.compile(loss='categorical_crossentropy', optimizer='Adam')
    #net.compile(loss='binary_crossentropy', optimizer=sgd)
    #xor.compile(loss="hinge", optimizer=sgd)
    #xor.compile(loss="binary_crossentropy", optimizer=sgd)
      
    #net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, show_accuracy=True, verbose=0)
#    history = net.fit_generator(X, Y, batch_size=batch_size, epochs = nrEpochs, verbose=1, validation_data=(Xv,Yv) )

    
    history=net.fit_generator(myGenerator(X= X,Y= Y), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
    

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(rootOutput + modelName + '_training_validation_acc_vs_epoch' + '.pdf')
#    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(rootOutput + modelName + '_training_validation_loss_vs_epoch' + '.pdf')
#    plt.show()


    if save_model_b == 1:
         
        json_string = net.to_json()
        open(rootOutput + modelFileName, 'w').write(json_string)
        net.save_weights(rootOutput + modelFileName + '.h5',overwrite=True)

    #test it. First read in the test data:

    #Read in the test data we avoid the chromos used for training:    
    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
    Xt,Yt, genomeSeqSourceTest = genSamples_II(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, flankSize = customFlankSize, 
                                            flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)        
        
    score, acc = net.evaluate(Xt, Yt, batch_size=batch_size, verbose=1)

    print('Test score:', score)
    print('Test accuracy:', acc)
    
    
    #Write run-data to txt-file for documentation of the run:
    runDataFileName = rootOutput + modelName + '_runData' + '.txt'
    runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
    
    s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
    s += modelDescription  + "\n"   
    s += 'Performance on test set, loss and accuracy resp.: ' + str(score) + ' ' + str(acc) + "\n"   
    if save_model_b == 1:
        s+= 'Model data obtained after training the model are recorded in: ' +  rootOutput + modelName + ' and ' + rootOutput + modelName + '.h5\n' 
    runDataFile.write(s)
    
    s = '' #reset
    runDataFile.write(s + "\n") #insert blank line
    #Which genome data were used?:
    if genSamples_b > 0.5:
        s = "Samples generated with python code from real genome." + "\n"
        s += "Genome data in file: " + genomeFileName + "\n"
        s += 'inclFrqModel_b: ' + str(1) + "\n"
        s += 'frqModelFileName: ' + frqModelFileName + "\n"  
        s += "Letters are one-hot encoded" + "\n"
        s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
        s += "trainDataInterval:" + str(trainDataInterval)  + "\n"    
        s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
        s += "testDataInterval:" + str(testDataInterval)  + "\n" 
        if onlyOneRandomChromo_b == 1:
            s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
            s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
            s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
            s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
        else:
            s += "Read in the whole genome sequence" + "\n"
        s += "shuffle_b = " + str(shuffle_b) + "\n"
        s += "inner_b = " + str(inner_b) + "\n"
        s += "shuffleLength = " + str(shuffleLength) +  "\n"
        s += "nrTrainSamples:" + str(nrTrainSamples)  + "\n"
        s += "trainDataInterval:" + str(trainDataInterval)  + "\n"  
        s += "nrTestSamples:" + str(nrTestSamples)  + "\n"
        s += "testDataInterval:" + str(testDataInterval)  + "\n" 
     
    elif genSamplesFromRandomGenome_b > 0.5:
        
        s = "Samples from random genome, all generated with python code." + "\n"
        s += "Genome data in file: " + randomGenomeFileName
            
    runDataFile.write(s)
    
    s = '' #reset
    runDataFile.write(s + "\n") #insert blank line
    #various params:    
    s= 'loss = "categorical_crossentropy"\n'       
    s += 'trainDataInterval: ' + str(trainDataInterval) + "\n"
    s += 'testDataInterval: ' + str(testDataInterval) + "\n" 
    s += 'customFlankSize_b: ' + str(customFlankSize_b) + "\n" 
    s += 'customFlankSize: ' + str(customFlankSize) + "\n" 
    s += 'genSamples_b: ' + str(genSamples_b) + "\n" 
    s += 'genomeFileName: ' + genomeFileName + "\n" 
    s += "outputEncodedOneHot_b: " + str(1) + "\n" 
    s += "outputEncodedInt_b: " + str(0) + "\n" 
    s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n" 
    s += "avoidChromo: " + str(avoidChromo)  + "\n" 
    s += 'genSamplesFromRandomGenome_b: ' + str(genSamplesFromRandomGenome_b) + "\n" 
    s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
    s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
    s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
    s += 'Optimizer: ' + optimizer + "\n"
    s += 'learning rate: ' + str(learningRate)  + "\n" 
    s += 'momentum: ' + str(momentum) + "\n" 
    s += 'batchSize: ' + str(batchSize) + "\n"
    s += 'dropoutVal: ' + str(dropoutVal) + "\n"
    s += 'dropoutLastLayer_b: ' + str(dropoutLastLayer_b) + "\n"
    s += 'padding: ' + padding + "\n"
    s += 'pool_b: ' +  str(pool_b) + "\n"
    s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
    s += 'poolAt: ' +  str(poolAt) + "\n"
    s += "dynSamplesTransformStyle_b:" + str(dynSamplesTransformStyle_b) + "\n"
    s += 'stepsPerEpoch: ' + str(stepsPerEpoch) + "\n" 
    s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
    s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
    s += 'letterShape: ' + str(letterShape) + "\n" 
    s += 'save_model_b: ' + str(save_model_b) + "\n" 
    s += 'modelName: ' + modelName + "\n" 
    s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
    
    runDataFile.write(s)
        
    #Params for net:
    s = '' #reset
    runDataFile.write(s + "\n") #insert blank line
    s = 'lengthWindows: ' + str(lengthWindows)  + "\n" 
    s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
    s += 'nrFilters: ' + str(nrFilters)
    
    runDataFile.write(s)
    
    runDataFile.close()
    

#######################################################################################
    
########### FINE
    
#######################################################################################
        
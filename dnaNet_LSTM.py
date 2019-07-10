# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk
"""

'''
Notes:
    -- first version extracted from dnaNet_v7
    -- contains only code for LSTM models

    
Usage:
    
In general: all functions called "allInOne"-something include/call all what is needed for training/validation fo a particular model.
So allInOneWithDynSampling_ConvLSTMmodel, will train/test a LSTM model which uses a convolutional to take care of "word embedding";
the function calls the code for building the model, for compiling it and for dynamically sampling from the desired data. The 
parameters of the function allow to specify the model, the sampling and the training.
    

####################################################

Input data:

    ####################################################

# Human genome 

#rootGenome = r"/Users/newUser/Documents/clouds/Sync/Bioinformatics/various_python/DNA_proj/data/human/"

#On binf servers:
#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName


#rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName



####################################################

#Set up training schedule, model and run:

####################################################

#For looking at the flow:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 0 #loads in model from repeatNr 115!
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 1
batchSize = 10
stepsPerEpoch = 10
trainDataIntervalStepSize = 200
trainDataInterval = [20000,21000]
nrTestSamples = 10
testDataInterval = [21000,22000]


#For testing a set-up (check taht it runs, how long training time is ...):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 0
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 10
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 2000000
trainDataInterval = [00000000,10000000]
nrTestSamples = 500000
testDataInterval = [10000000,12000000]

#In anger:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 500
firstRepeatNr = 116 #loads in model from repeatNr 115!
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,12000000]



#Modelling spec's
exonicInfoBinaryFileName  = ''
inclFrqModel_b = 0
insertFrqModel_b = 0
customFlankSize = 50
overlap = 0
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [4]
nrFilters = [256] 
filterStride = 1
#parallel LSTMs:
nrOfParallelLSTMstacks = 1
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [20]
#Nr of lstm layers:
nrLSTMlayers = 2
padding = 'valid'


#set-up
dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/ßroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
file = "frqModel_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
exclFrqModelFlanks_b = 0
augmentWithRevComplementary_b = 0
dropout_b = 0
dropoutVal = 0.0
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.1 #default, but we use Adam here, so the value here isn't used
#learningRate = learningRate
onlyOneRandomChromo_b = 0
avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1 


subStr = '_forModelPlot_1Conv2LayerLstm_flanks50_win4_stride1_overlap0_dropout00'
learningRate = 0.001
modelName = 'ownSamples/human/inclRepeats/modelLSTM_' + subStr
modelDescr = subStr


#With conv layer:
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#Only LSTM:
labelsCodetype = 1 #1: base pair prediction
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, convLayers_b = 0, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,   lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

'''



#THEANO_FLAGS='floatX=float32,device=cuda' 
#TENSORFLOW_FLAGS='floatX=float32,device=cuda' 

import os
import argparse

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

#import frqModels as frqM

import pickle

#import graphviz
#import pydot



from dnaNet_dataGen import * #all smpling aso is here


###############################
## LSTM's
###############################

def makeLSTMmodel(sequenceLength, nrLayers = 1, letterShape = 4, outputSize = 4, batchSize = 100 , return_sequences=False, stateful=False, dropout_b = 0, dropoutVal = 0.25):
    '''
    network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    outputSize = the size of the output layer, here 4
    '''

    print('Build LSTM model...')

    inputs_left = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize,sequenceLength,letterShape))
    inputs_right = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize, sequenceLength,letterShape))

#    #all but the last lstm -layer returns a sequence (of the same length as the input)
#    for i in range(nrLayers -1):
#
#        inputs_left  = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_left)
#        inputs_right = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_right)
#
#        print(inputs_left._keras_shape)
#        print (inputs_right._keras_shape)
    
    if nrLayers > 1: 
    
        lstm_left_1  = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_left)
        lstm_right_1 = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_right)
    
        print(lstm_left_1._keras_shape)
        print (lstm_right_1._keras_shape)
    
        if dropout_b ==  1:
            
                lstm_left_1 = Dropout(dropoutVal)(lstm_left_1)
                lstm_right_1 = Dropout(dropoutVal)(lstm_right_1)
            

    if nrLayers > 2: 
    
        lstm_left_2  = LSTM(letterShape, return_sequences=True, stateful=stateful)(lstm_left_1)
        lstm_right_2 = LSTM(letterShape, return_sequences=True, stateful=stateful)(lstm_right_1)
    
        print(lstm_left_2._keras_shape)
        print (lstm_right_2._keras_shape)
    
        if dropout_b ==  1:
            
                lstm_left_2 = Dropout(dropoutVal)(lstm_left_2)
                lstm_right_2 = Dropout(dropoutVal)(lstm_right_2)


    if nrLayers > 3: 

        lstm_left_3  = LSTM(letterShape, return_sequences=True, stateful=stateful)(lstm_left_2)
        lstm_right_3 = LSTM(letterShape, return_sequences=True, stateful=stateful)(lstm_right_2)
    
        print(lstm_left_3._keras_shape)
        print (lstm_right_3._keras_shape)
    
        if dropout_b ==  1:
            
                lstm_left_3 = Dropout(dropoutVal)(lstm_left_3)
                lstm_right_3 = Dropout(dropoutVal)(lstm_right_3)


    #Define the output of the first part:
    if nrLayers == 1:

        left_firstPart = inputs_left
        right_firstPart = inputs_right
        
    elif nrLayers == 2:
        
        left_firstPart = lstm_left_1
        right_firstPart = lstm_right_1

    elif nrLayers == 3:
        
        left_firstPart = lstm_left_2
        right_firstPart = lstm_right_2
        
    elif nrLayers == 4:
        
        left_firstPart = lstm_left_3
        right_firstPart = lstm_right_3
    
    #last lstm layer:        
    lstm_left  = LSTM(letterShape, return_sequences=False, stateful=stateful)(left_firstPart)
    lstm_right = LSTM(letterShape, return_sequences=False, stateful=stateful)(right_firstPart)

    print("Left-hand shape after first LSTM part ", lstm_left._keras_shape)
    print("Right-hand shape after first LSTM part ",lstm_right._keras_shape)

    
    #Concatenate the two LSTM-outputs:
    leftAndRight = concatenate([lstm_left, lstm_right], axis=-1, name = 'concat')
    
    print(leftAndRight._keras_shape)


    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print(prediction._keras_shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)


    print("... build model.")
        
    return model



def makeConv1DLSTMmodel(sequenceLength, letterShape, lengthWindows, nrFilters, filterStride = 1, onlyConv_b = 0, nrOfParallelLSTMstacks = 1, finalDenseLayers_b = 0, sizeHidden = [10], paddingType = 'valid', outputSize = 4,  batchSize = 100, pool_b = 0, maxPooling_b = 0, poolAt = [2], dropoutConvLayers_b = 1, dropoutVal = 0.25, return_sequences=False, stateful=False):
    '''
    network model
    flankSize = lenght of the flanking sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the final dense layer (if finalDenseLayer_b = 1)
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv1d plus LSTM model...')    
    

#    inputs_left = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize,sequenceLength,letterShape))
#    inputs_right = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize, sequenceLength,letterShape))

    inputs_left = Input(shape=(sequenceLength,letterShape))
    inputs_right = Input(shape=(sequenceLength,letterShape))

    print("First the 1d-convo, ba ...")

    convOutLeft = Conv1D(kernel_size=lengthWindows[0], strides=filterStride, filters=nrFilters[0], padding=paddingType, activation='relu')(inputs_left)
    convOutRight = Conv1D(kernel_size=lengthWindows[0], strides=filterStride, filters=nrFilters[0], padding=paddingType, activation='relu')(inputs_right)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            convOutLeft = MaxPooling1D(convOutLeft)
            convOutRight = MaxPooling1D(convOutRight)
        else:
            convOutLeft = AveragePooling1D(convOutLeft)
            convOutRight = AveragePooling1D(convOutRight)

    if dropoutConvLayers_b ==  1:
        
        convOutLeft = Dropout(dropoutVal)(convOutLeft)
        convOutRight = Dropout(dropoutVal)(convOutRight)
        
    print("Left-hand shape after 1st convo ", convOutLeft._keras_shape)
    print("Right-hand shape after 1st convo ",convOutRight._keras_shape)

    
    for i in range(len(nrFilters)-1):    
    
        convOutLeft = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutLeft)
        convOutRight = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutRight)

        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                convOutLeft = MaxPooling1D(convOutLeft)
                convOutRight = MaxPooling1D(convOutRight)
            else:
                convOutLeft = AveragePooling1D(convOutLeft)
                convOutRight = AveragePooling1D(convOutRight)
                
                
        if dropoutConvLayers_b ==  1:
        
            convOutLeft = Dropout(dropoutVal)(convOutLeft)
            convOutRight = Dropout(dropoutVal)(convOutRight)
            
        print(convOutLeft._keras_shape)
        print(convOutRight._keras_shape)


    print("...by!")
    
    
    
    print("Left-hand shape after all convo's ",convOutLeft._keras_shape)
    print("Right-hand shape after all convo's ", convOutRight._keras_shape)
    
    
    if onlyConv_b == 1:
        
        flattenLeft = Reshape((-1,))(convOutLeft)
        print(flattenLeft._keras_shape)
        flattenRight = Reshape((-1,))(convOutRight)
        print(flattenRight._keras_shape)
        leftAndRight = concatenate([flattenLeft, flattenRight], axis = -1) 
        
    else:
        
        print("Then the lstm part ..." )
    
    #    lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
    #    lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
    #
    #    print(lstm_left_1._keras_shape)
    #    print (lstm_right_1._keras_shape)
    #
    #    #Concatenate the two LSTM-outputs:
    #    leftAndRight = concatenate([lstm_left_1, lstm_right_1], axis=-1)
    
        for j in range(nrOfParallelLSTMstacks):
    
            lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutLeft)
            lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutRight)
        
            print("Left-hand shape after 1st LSTM ", lstm_left_1._keras_shape)
            print("Right-hand shape after 1st LSTM ",lstm_right_1._keras_shape)
            
        #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)
        #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
        #
        #    print(lstm_left_2._keras_shape)
        #    print (lstm_right_2._keras_shape)
        
        
            lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_left_1)
            lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_right_1)
            
        #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
        #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
            
        
            print("Left-hand shape after 2nd LSTM ",lstm_left_2._keras_shape)
            print ("Right-hand shape after 2nd LSTM ", lstm_right_2._keras_shape)
            
            #Concatenate the two LSTM-outputs:
            leftAndRight_j = concatenate([lstm_left_2, lstm_right_2], axis=-1)
            
            if j == 0:
                
                leftAndRight = leftAndRight_j
                
            else: 
                
                leftAndRight = concatenate([leftAndRight, leftAndRight_j], axis = -1) 
                
        
            print("Shape of concatenated LSTM output ", leftAndRight._keras_shape)
            
        
        print("Shape of LSTM-stacks output ", leftAndRight._keras_shape)
    
    if finalDenseLayers_b == 1:
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight._keras_shape)
    
    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print("Output shape ", prediction._keras_shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)

    print("... Model's build.")
    
    return model



#to define layer in makeConv1DLSTMmodelFusedWithEROmodel, which consists in 
#scaling the output tensors from parallel LSTM by the output of an ExonicRepeatOther-model
#and summing the components:
def scaleAndAdd(inputList):
    

    t1 = inputList[0]
    t2 = inputList[1]
#    print(t1._keras_shape, t2._keras_shape )
#    print t1[0], t2[0]
#    return backend.sum([t1[0][0]*t2[0][0], t1[0][1]*t2[0][1], t1[0][2]*t2[0][2]])

#    print t1, t2

#    return backend.sum(t1*t2, axis = -1, keepdims =True)
#    return backend.sum([t1[0]*t2[0], t1[1]*t2[1], t1[2]*t2[2]], axis = 1)
#    return tf.tensordot(t1,t2, axes = 0) 

#    output = backend.placeholder(shape = 512) #t2._keras_shape) 
#
#    return  t1[0][0]*t2[0][1]

#    return output

    return t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2]


def transpose(t):
    
    from keras import backend 
    
    return backend.transpose(t)


def summation(t):
    
    from keras import backend 
    
    return backend.sum(t, axis = 1, keepdims = False)


def makeConv1DLSTMmodelFusedWithEROmodel(eroModel, sequenceLength, letterShape, lengthWindows, nrFilters, filterStride = 1, nrOfParallelLSTMstacks = 1, finalDenseLayers_b = 0, sizeHidden = [10], paddingType = 'valid', outputSize = 4,  batchSize = 100, pool_b = 0, maxPooling_b = 0, poolAt = [2], dropoutConvLayers_b = 1, dropoutVal = 0.25, return_sequences=False, stateful=False):
    '''
    network model
    
    '''

    print('Build Conv1d plus LSTM model...')    
    

#    inputs_left = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize,sequenceLength,letterShape))
#    inputs_right = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize, sequenceLength,letterShape))

    inputs_left = Input(shape=(sequenceLength,letterShape))
    inputs_right = Input(shape=(sequenceLength,letterShape))

    print("First the 1d-convo, ba ...")

    convOutLeft = Conv1D(kernel_size=lengthWindows[0], strides=filterStride, filters=nrFilters[0], padding=paddingType, activation='relu')(inputs_left)
    convOutRight = Conv1D(kernel_size=lengthWindows[0], strides=filterStride, filters=nrFilters[0], padding=paddingType, activation='relu')(inputs_right)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            convOutLeft = MaxPooling1D(convOutLeft)
            convOutRight = MaxPooling1D(convOutRight)
        else:
            convOutLeft = AveragePooling1D(convOutLeft)
            convOutRight = AveragePooling1D(convOutRight)

    if dropoutConvLayers_b ==  1:
        
        convOutLeft = Dropout(dropoutVal)(convOutLeft)
        convOutRight = Dropout(dropoutVal)(convOutRight)
        
    print("Left-hand shape after 1st convo ", convOutLeft._keras_shape)
    print("Right-hand shape after 1st convo ",convOutRight._keras_shape)

    
    for i in range(len(nrFilters)-1):    
    
        convOutLeft = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutLeft)
        convOutRight = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutRight)

        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                convOutLeft = MaxPooling1D(convOutLeft)
                convOutRight = MaxPooling1D(convOutRight)
            else:
                convOutLeft = AveragePooling1D(convOutLeft)
                convOutRight = AveragePooling1D(convOutRight)
                
                
        if dropoutConvLayers_b ==  1:
        
            convOutLeft = Dropout(dropoutVal)(convOutLeft)
            convOutRight = Dropout(dropoutVal)(convOutRight)
            
        print(convOutLeft._keras_shape)
        print(convOutRight._keras_shape)


    print("...by!")
    
    
    
    print("Left-hand shape after all convo's ",convOutLeft._keras_shape)
    print("Right-hand shape after all convo's ", convOutRight._keras_shape)
        
    
    print("Then the lstm part ..." )
    
    lstmOutputList = []

#    lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
#    lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
#
#    print(lstm_left_1._keras_shape)
#    print (lstm_right_1._keras_shape)
#
#    #Concatenate the two LSTM-outputs:
#    leftAndRight = concatenate([lstm_left_1, lstm_right_1], axis=-1)

    for j in range(nrOfParallelLSTMstacks):

        lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutLeft)
        lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutRight)
    
        print("Left-hand shape after 1st LSTM ", lstm_left_1._keras_shape)
        print("Right-hand shape after 1st LSTM ",lstm_right_1._keras_shape)
        
    #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)
    #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
    #
    #    print(lstm_left_2._keras_shape)
    #    print (lstm_right_2._keras_shape)
    
    
        lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_left_1)
        lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_right_1)
        
    #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
    #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
        
    
        print("Left-hand shape after 2nd LSTM ",lstm_left_2._keras_shape)
        print ("Right-hand shape after 2nd LSTM ", lstm_right_2._keras_shape)
        
        #Concatenate the two LSTM-outputs:
        leftAndRight_j = concatenate([lstm_left_2, lstm_right_2], axis=-1)

        if j == 0:

            lstmOutputList = leftAndRight_j
            
        else:             
           
            lstmOutputList = concatenate([lstmOutputList, leftAndRight_j], axis = 1)
            
        print("iter, shape:", j,lstmOutputList._keras_shape)
#        lstmOutputList.append(leftAndRight_j)        

#    lstmOutputList = Input(lstmOutputList)
#
#    lstmOutputList = Reshape(( leftAndRight_j._keras_shape[0], nrOfParallelLSTMstacks, leftAndRight_j._keras_shape[1] ), input_shape =  lstmOutputList._keras_shape)(lstmOutputList)

    newSize = int(lstmOutputList._keras_shape[1]/nrOfParallelLSTMstacks) 
    print("newSize ", newSize)

    lstmOutputList = Reshape((  nrOfParallelLSTMstacks, newSize ), input_shape =  lstmOutputList._keras_shape)(lstmOutputList) #convOutLeft._keras_shape[1] repl'ed 512

    print (lstmOutputList._keras_shape)


    print("Next get the ERO output ..")
    
#    eroOutput = Model((inputs_left, inputs_right), eroModel.output)
    eroOutput = eroModel([inputs_left, inputs_right])
    print (eroOutput._keras_shape, eroOutput.dtype)
    
    print(eroOutput, [eroOutput], "ERO ouput")
        
    print(".. and scale the LSTMs with the ERO output:")
#    scaleAndAddLayer = Lambda(function = scaleAndAdd, output_shape =  leftAndRight_j._keras_shape)
#    mergedInput = tf.concat([eroOutput, lstmOutputList], axis=0)
#    print (mergedInput._keras_shape, mergedInput.dtype)
#    leftAndRight = scaleAndAddLayer([eroOutput[0], lstmOutputList[0]])
    
    eroOutputRepeated = RepeatVector(newSize)(eroOutput) #newSize repl'ed 512
    print (eroOutputRepeated._keras_shape)
#    x = eroOutputRepeated.eval(session = sess)
#    tf.Print(x)
    #this should perform a transpose:
    eroOutputRepeatedT = Permute((2, 1), input_shape = eroOutputRepeated._keras_shape)(eroOutputRepeated)
    print (eroOutputRepeatedT._keras_shape)
#    y = eroOutputRepeatedT.eval(session = sess)
#    tf.Print(y)
    
    tf.Print(eroOutputRepeatedT, [eroOutputRepeatedT], "ERO ouput, repeated and transposed")
    
#    mergedInput = Input([eroOutputRepeatedT, lstmOutputList])
    leftAndRight = multiply([eroOutputRepeatedT, lstmOutputList])
        
#    leftAndRight = Add()(leftAndRight)
                
    print("Shape after multiply-layer ", leftAndRight._keras_shape)

    leftAndRight = Lambda(function = summation, output_shape = (512,))(leftAndRight)
    
#    leftAndRight = Reshape((512 ), input_shape =  leftAndRight._keras_shape)(leftAndRight)
    
    print("Shape of resulting output ", leftAndRight._keras_shape)
    
    if finalDenseLayers_b == 1:
        
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight._keras_shape)
    
    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print("Output shape ", prediction._keras_shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)

    print("... Model's build.")
    
    return model





def allInOneWithDynSampling_ConvLSTMmodel(nrOuterLoops = 1,
                                          firstIterNr = 0,
                                          nrOfRepeats = 1,
                                          firstRepeatNr = 0,
                                          loss = "categorical_crossentropy", 
                                          usedThisModel = 'makeConv1DLSTMmodel', #set this manually if restarting
                                          onHecaton_b = 0,
                                          convLayers_b = 1,
                                          onlyConv_b = 0,
                                          leftRight_b = 1,
                                          fusedWitEROmodel_b = 0,
                                          eroModelFileName = '',
                                          nrLSTMlayers = 2, 
                                          nrOfParallelLSTMstacks = 1,
                                          finalDenseLayers_b = 0, #if 1 set the hiddenUnits param
                       learningRate = 0.01,
                       momentum = 0.0, 
            trainDataIntervalStepSize = 100000, 
            trainDataInterval0 = [0,200000] ,
            testDataIntervalIdTotrainDataInterval_b = 0,
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000],  #not used
            customFlankSize_b = 1,                       
            customFlankSize = 50,
            overlap = 0, 
            genSamples_b = 1, 
            genomeFileName = '',
            exonicInfoBinaryFileName  = '',
            outputEncodedOneHot_b = 1,
            labelsCodetype = 0,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
#            genSamplesFromRandomGenome_b = 0, #KEEP THIS
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            getOnlyRepeats_b = 0,
            augmentWithRevComplementary_b = 0, 
            augmentTestDataWithRevComplementary_b = 0, 
            inclFrqModel_b = 0,
            insertFrqModel_b = 0,
            frqModelFileName = '',
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0,
            optimizer = 'ADAM',
            batchSize = 128, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=4,
            letterShape = 4, # size of the word            
            lengthWindows = [3, 6],
            hiddenUnits= [50], #for conv1d and conv2d only the first entry is used 
            dropout_b = 0,
            dropoutVal= 0.25,
            dropoutLastLayer_b = 0,
            nrFilters = [200, 100],
            filterStride = 1,
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
            modelDescription = 'LSTM type ... to be filled in!',
            on_binf_b = 1, 
            testOnly_b = 0,
            path = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj"):
                
    if on_binf_b == 1:
        root = path + r"/Inputs/"
        rootDevelopment = path + r"/development/"
        rootOutput = path + r"/results_nets/"
    else:
        path = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/"
        root = path + r"/Inputs/"
        rootDevelopment = path + r"/development/"
        rootOutput = path + r"/results_nets/"
    
    '''
        labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding and not including the frq model 
                (inclFrqModel_b = 0).                
    '''
                
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
                
                else: 
                    
                    sizeInput = 2*customFlankSize + 1 
                                                        
            else:  
            
                sizeInput = 2*customFlankSize 

                #If augmentWithRevComplementary_b = 0, batchSize = Xconv.shape[0]; if = 1 we get back twice batchSize:
                if augmentWithRevComplementary_b == 0:
                    Xleft = np.zeros(shape = (batchSize, customFlankSize + overlap, letterShape))
                    Xright = np.zeros(shape = (batchSize, customFlankSize + overlap, letterShape))
                else:
                    Xleft = np.zeros(shape = (2*batchSize, customFlankSize + overlap, letterShape))
                    Xright = np.zeros(shape = (2*batchSize, customFlankSize + overlap, letterShape))
    
            print("sizeInput is set to: ", sizeInput)
            
            
            
            #we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so; the data for testing is loaded after
            #the training is done (below) so as to avoid spending the memory needed for the test data during 
            #the training part: 
            frqModelDict = {}
            if inclFrqModel_b == 1:
                
                frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)
                                         
                                
                
    
#            #we fetch the output from the frq model if we want to include it in the training and testing; 
#            #the test set shall also includes the frq model output if so: 
#            frqModelDict = {}
#            if inclFrqModel_b == 1:
#                
#                frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)
#    
#                #Read in the test data we avoid the chromos used for training:    
#                avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
#                Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, flankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b,
#                                                        flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
#    
#                if insertFrqModel_b != 1:
#                    
#                    #Split the test data as the training data:
#                    nrOfTestSamples = Xt.shape[0]
#                    Xconv_t = np.zeros(shape = (nrOfTestSamples, sizeInputConv, letterShape))
#                    Xfrq_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
#        
#                    Xconv_t[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
#                    Xconv_t[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
#                    Xfrq_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
#        
#        #            XsplitList_t = []            
#        #            for i in range(nrOfTestSamples):
#        #                
#        #                XsplitList_t.append([Xfrq_t[i], Xconv_t[i]])
#        #            
#        #            Xsplit_t = np.asarray(XsplitList_t)
#                        
#                        
#        #            print Xconv_t.shape, Xfrq_t.shape            
#                
#            else:
#    
#                #Read in the test data we avoid the chromos used for training:    
#                avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
#                Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
#    
#                #If augmentWithRevComplementary_b = 0, nrTestSamples = Xt.shape[0]; if = 1 we get back twice nrTestSamples, but still Xt.shape[0]:
#                Xt_left = np.zeros(shape = (Xt.shape[0], customFlankSize + overlap, letterShape))
#                Xt_right = np.zeros(shape = (Xt.shape[0], customFlankSize + overlap, letterShape))
#    
#    
#                print "Xt shape", Xt.shape
#                print "Xt_left shape, Xt_right shape", Xt_left.shape, Xt_right.shape
#    
#    
##                Xt_left[:, :(customFlankSize + overlap), :] = Xt[:, :(customFlankSize + overlap), :]
#                Xt_left = Xt[:, :(customFlankSize + overlap), :].copy()
#                Xt_right = Xt[:, (customFlankSize - overlap):, :].copy()
#                #and reverse it:
#                Xt_right = np.flip(Xt_right, axis = 1)
#                    
                         
            
            #Dynamically fetch small sample batches; this runs in an infinite loop
            #in parallel to the fit_generator call below (and stops when that is done)
            def myGenerator(customFlankSize,batchSize, inclFrqModel_b, insertFrqModel_b, labelsCodetype):
               
                while 1:
                    X,Y = genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                     genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
    #                sizeInput = X.shape[1]
                    if inclFrqModel_b == 1  and insertFrqModel_b != 1:
            
                        Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                        Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                        Xfrq[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
                        
                        yield([Xfrq, Xconv],Y)
                        
                    
                    elif onlyConv_b == 1 and leftRight_b == 0:
                        
                        yield(X,Y)
                    
                    else:
                        
#                        Xleft[:, :(customFlankSize+overlap), :] = X[:, :(customFlankSize + overlap) , :]
                        Xleft = X[:, :(customFlankSize + overlap) , :].copy()
                        Xright = X[:, (customFlankSize - overlap):, :].copy()
                        #and reverse it:
                        Xright = np.flip(Xright, axis = 1)
    
                        yield([Xleft, Xright], Y)
                        
#                        print "X left shape", Xleft.shape
#                        print "X right shape", Xright.shape

                    
        #when if augmentWithRevComplementary_b == 1 the generated batches contain 2*batchSize samples:
        if augmentWithRevComplementary_b == 0:

            batchSizeReal = batchSize

        else:

            batchSizeReal = 2*batchSize
    

        #output size depends on what we want to predict (base or base pair or pyri/puri)
        if labelsCodetype == 0:
            sizeOutput = 4
        elif labelsCodetype == 1 or labelsCodetype == -1 or labelsCodetype == 3:
            sizeOutput = 2
        elif labelsCodetype == 2:
            sizeOutput = 3
    
    
        if testOnly_b == 0: #just means that we're running a regular trianing/testing session
    
            #Write run-data to txt-file for documentation of the run:
            runDataFileName = modelFileName + '_runData.txt'
            runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
            
            s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
            s += modelDescription  + "\n"  
            s += 'ExonRepaetOther (ERO) prediction model included?: ' + str(fusedWitEROmodel_b) + "\n"  
            s += 'eroModelFileName: ' + eroModelFileName + "\n" 
            if save_model_b == 1:
                s+= 'Model data obtained after training the model are recorded in: ' +  modelFileName + ' and ' + rootOutput + modelName + '.h5\n' 
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
            #Which genome data were used?:
            if genSamples_b > 0.5:
                s = "Samples generated with python code from real genome." + "\n"
                s += "Genome data in file: " + genomeFileName + "\n"
                s += "exonicInfoBinaryFileName: " + exonicInfoBinaryFileName + "\n"
                s += "Letters are one-hot encoded" + "\n"
                s += "Labels are encoded as type" + str(labelsCodetype) + "\n"
                if onlyOneRandomChromo_b == 1:
                    s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                    s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
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
            s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
            s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
            s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
            s += 'learningRate: ' + str(learningRate) + "\n"
            s += 'batchSize: ' + str(batchSize) + "\n"
            s += 'dropout_b: ' + str(dropout_b) + "\n"
            s += 'dropoutVal: ' + str(dropoutVal) + "\n"
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
            s = 'convLayers_b: ' + str(convLayers_b) + "\n"
            s += 'lengthWindows: ' + str(lengthWindows)  + "\n" 
            s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
            s += 'nrFilters: ' + str(nrFilters)  + "\n" 
            s += 'filterStride: ' + str(filterStride)  + "\n" 
            s += 'nrLSTMlayers: ' + str(nrLSTMlayers)  + "\n" 
            s += 'nrOfParallelLSTMstacks: ' + str(nrOfParallelLSTMstacks)
        
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
        
            runDataFile.close()
            #Write run-data to txt-file for documentation of the run: DONE


        #Run series of repeated training-and-testing sessions each consisting in nrEpochs rounds:
        for k in range(firstRepeatNr, nrOfRepeats):       
                    
    
            #in first outer-iteration build the model; thereafter reload the latest stored version (saved below)
            if n == 0 and k == 0: 
        
                if convLayers_b == 0 and fusedWitEROmodel_b == 0:
            
                    net = makeLSTMmodel(sequenceLength = customFlankSize + overlap, nrLayers = nrLSTMlayers, letterShape = letterShape,  outputSize = sizeOutput, batchSize = batchSizeReal, dropout_b = dropout_b, dropoutVal = dropoutVal )
                    
                    usedThisModel = 'makeLSTMmodel'
                    
                elif convLayers_b > 0 and fusedWitEROmodel_b == 0:
                    
                    if onlyConv_b != 1 or (onlyConv_b == 1 and leftRight_b == 1):
                    
                        net = makeConv1DLSTMmodel(sequenceLength = customFlankSize + overlap, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, filterStride = filterStride, onlyConv_b = onlyConv_b, nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, finalDenseLayers_b = finalDenseLayers_b, sizeHidden = hiddenUnits, outputSize = sizeOutput,  batchSize = batchSizeReal, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutConvLayers_b = dropout_b, dropoutVal = dropoutVal )
                
                        usedThisModel = 'makeConv1DLSTMmodel'
                    
                    elif onlyConv_b == 1 and leftRight_b != 1:
                    
                        net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b)
        
                        usedThisModel = 'makeConv1Dmodel'
                    
                elif fusedWitEROmodel_b == 1:
                    
                    eroModel = model_from_json(open(eroModelFileName).read())
                    eroModel.load_weights(eroModelFileName +'.h5')
                    
                    net = makeConv1DLSTMmodelFusedWithEROmodel(eroModel = eroModel, sequenceLength = customFlankSize + overlap, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, filterStride = filterStride, nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, finalDenseLayers_b = finalDenseLayers_b, sizeHidden = hiddenUnits, outputSize = sizeOutput,  batchSize = batchSizeReal, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutConvLayers_b = dropout_b, dropoutVal = dropoutVal )
                    
                    usedThisModel = 'makeConv1DLSTMmodelFusedWithEROmodel'
                    
                    
                #Call a summary of the model:
                net.summary()
                #Save a plot of the model:
                if onHecaton_b == 0:
                    plot_model(net, to_file= rootOutput + modelName + '_plot.png', show_shapes=True, show_layer_names=True)
                
                    
            else:
    
                #reload the model from previous iter/repeat
                if k == 0:
                    modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n-1) + '_repeatNr' + str(nrOfRepeats-1)
                else:
                    modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n) + '_repeatNr' + str(k-1)
                    
                net = model_from_json(open(modelFileNamePrevious).read())
                net.load_weights(modelFileNamePrevious +'.h5')
        
                print("I've now reloaded the model from the previous iteration: ", modelFileNamePrevious)
                        

        
            print("Next: compile it .."     )
        
        
            if optimizer == 'SGD':
                
                optUsed = SGD(lr= learningRate, decay=1e-6, momentum= momentum, nesterov=True)
                #sgd = SGD(lr=0.01)
        
            elif optimizer =='ADAM':
            
                optUsed = Adam(lr= learningRate)
            
            elif optimizer == 'RMSprop':
            
                optUsed = RMSprop(lr=learningRate, decay = 1e-3)
        
                
            net.compile(loss=loss, optimizer=optUsed, metrics=['accuracy'])    
        
            print("Compiled model ..."    )
        

            if testOnly_b == 0:#just means that we're running a regular trianing/testing session

                history = net.fit_generator(myGenerator(customFlankSize,batchSize, inclFrqModel_b, insertFrqModel_b, labelsCodetype), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
           
    
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
    
         
    
                        
            #test it. First read in the test data:
            #If so desired, we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so: 
            if testDataIntervalIdTotrainDataInterval_b == 0:
               
                if inclFrqModel_b == 1:
                        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
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
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                              flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  
                                                              onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
        
                    #If augmentWithRevComplementary_b = 0, nrTestSamples = Xt.shape[0]; if = 1 we get back twice nrTestSamples, but still Xt.shape[0]:
                    Xt_left = np.zeros(shape = (Xt.shape[0], customFlankSize + overlap, letterShape))
                    Xt_right = np.zeros(shape = (Xt.shape[0], customFlankSize + overlap, letterShape))
        
        
                    print("Xt shape", Xt.shape)
                    print("Xt_left shape, Xt_right shape", Xt_left.shape, Xt_right.shape)
        
        
    #                Xt_left[:, :(customFlankSize + overlap), :] = Xt[:, :(customFlankSize + overlap), :]
                    Xt_left = Xt[:, :(customFlankSize + overlap), :].copy()
                    Xt_right = Xt[:, (customFlankSize - overlap):, :].copy()
                    #and reverse it:
                    Xt_right = np.flip(Xt_right, axis = 1)
                           

        
        
                if inclFrqModel_b == 1:
                    
                    if insertFrqModel_b == 1:
                        
                        score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
                    
                    else:
                        
                        score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
                else:
                    
                    score, acc = net.evaluate([Xt_left, Xt_right],Yt, batch_size=batchSizeReal, verbose=1)
                    
                    
            elif testDataIntervalIdTotrainDataInterval_b == 1: #we test using the dynamic sampling
                        
                score, acc = net.evaluate_generator(myGenerator(customFlankSize,batchSize, inclFrqModel_b, insertFrqModel_b, labelsCodetype), steps = np.int(float(nrTestSamples)/batchSize))
                    

                
#            #test it:
#            if inclFrqModel_b == 1 and insertFrqModel_b != 1:
#                score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
#            else:
#        
#                score, acc = net.evaluate([Xt_left, Xt_right],Yt, batch_size=batchSizeReal, verbose=1)
#        
#            print('Test score:', score)
#            print('Test accuracy:', acc)       
    
        
            print('Test score:', score)
            print('Test accuracy:', acc)
            
            #record and plot the total test performance, ie up to this iter/repeat:
            testHistoryTotal['acc'].append(acc)
            testHistoryTotal['loss'].append(score)
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
            

        if testOnly_b == 0:#just means that we're running a regular trianing/testing session     
            
            runDataFile = open(runDataFileName, 'a') #Obs: this will overwrite an existing file with the same name
            
            s = " " + "\n"  
            s += 'used this core model: ' + usedThisModel  + "\n" 
            if onlyOneRandomChromo_b == 1:
                s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
            s += 'Performance after outer iter ' + str(n) + ' on test set, loss and accuracy resp.: ' + str(score) + ' ' + str(acc) + "\n"               
            runDataFile.write(s)
            
            runDataFile.close()
            
   



#All in one run
def allInOneSampling_LSTMmodel(loss = "categorical_crossentropy", 
                       learningRate = 0.01,
            nrTrainSamples = 100000,
            trainDataInterval = [0,200000] , 
            nrValSamples = 20000,
            valDataInterval = [200000,400000],   
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000], 
            customFlankSize_b = 0, 
            customFlankSize = 50,
            genSamples_b = 1, 
            genomeFileName = '',
            outputEncodedOneHot_b = 1,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
#            genSamplesFromRandomGenome_b = 0, #KEEP THIS
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            augmentWithRevComplementary_b = 0, 
            batchSize = 128, 
            nrEpochs = 100,
            sizeOutput=4,
            letterShape = 4, # size of the word
            pool_b = 0,
            maxPooling_b = 0,
            poolAt = [],
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'LSTM type ... to be filled in!',
            on_binf_b = 1,
            path = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj"):
                
    if on_binf_b == 1:
        root = path + r"/Inputs/"
        rootDevelopment = path + r"/development/"
        rootOutput = path + r"/results_nets/"
    else:
        path = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/"
        root = path + r"/Inputs/"
        rootDevelopment = path + r"/development/"
        rootOutput = path + r"/results_nets/"


#1st version parameters:    
    lengthWindows = 3, 3, 4
    nrFilters = 20, 30, 40
    nrHiddenUnits = 10

  
    if genSamples_b > 0.5: #generate a set of random data acc to the sizes set

        #if a genomeFileName is specified, use that genome:
        if len(genomeFileName) > 0:
            fromGenome_b = 1
        else:
            fromGenome_b = 0
    
        X,Y, genomeSeqSourceTrain = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, genRandomSamples_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTrainSamples, startAtPosition = trainDataInterval[0], endAtPosition = trainDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        sizeInput = X.shape[1]
        print("Shape train set: ", X.shape)
        print("Shape train set labels: ", Y.shape)
        print("sizeInput: %d " % sizeInput)

        avoidChromo.append(genomeSeqSourceTrain) #to avoid getting val data from the same chromo as the training data 
        Xv,Yv, genomeSeqSourceVal = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrValSamples, startAtPosition = valDataInterval[0], endAtPosition = valDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        
        avoidChromo.append(genomeSeqSourceVal) ##to avoid getting test data from the same chromo as the training and validation data 
        Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        

#    elif genSamplesFromRandomGenome_b > 0.5: #generate a set of random data acc to the sizes set
#
#        #generate random genome of set size:   
#        genRandomGenome(length = randomGenomeSize, fileName = root + randomGenomeFileName, on_binf_b = on_binf_b) #will write the generated genome sequence to the file  
#
#        X,Y = genSamples_I(nrSamples = trainDataInterval[1] - trainDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
#        sizeInput = X.shape[1]
#
#        Xv,Yv = genSamples_I(nrSamples = valDataInterval[1] - valDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
#
#        Xt,Yt = genSamples_I(nrSamples = testDataInterval[1] - testDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        

    else:
        
        if outputEncodedOneHot_b == 1: #fetch the data from an appropriate source

            #Using the getData2-fct to fetch data:  
            fname=root + r"training.dat"
            vname = root + r"validation.dat"
            tname=root + r"test.dat"
        
            
            X,Y = getData2(fname, letterShape, sizeOutput, loadRecsInterval = trainDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
            sizeInput = X.shape[1]
                    
            Xv,Yv = getData2(vname, letterShape, sizeOutput, loadRecsInterval = valDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)      
                
            Xt,Yt = getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
        
    
    batch_size = min(batchSize,max(1,len(X)/20))
    print("batch size, nr of epocs: ", batch_size, nrEpochs)
      
    net = makeLSTMmodel(sequenceLength = sizeInput, hiddenUnits = nrHiddenUnits, letterShape = letterShape,  outputSize = sizeOutput, batchSize = batch_size)
#    net = makeConv1DLSTMmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, sizeHidden = nrHiddenUnits, outputSize = sizeOutput, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt)


    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=0.01)

    print("Next: compile it .."     )

    optimizer = RMSprop(lr=learningRate, decay = 1e-3)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    

    print("Compiled model ..."    )

#    net.fit(X, Y, batch_size=batch_size, nb_epochs=nrEpochs, verbose=0)
    history = net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, verbose=1, validation_data=(Xv,Yv) )

#    for i in range(nrEpochs):
#        history = net.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(Xv,Yv), shuffle=False )
#        net.reset_states()
#    
    #test it:
    score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
#    net.reset_states()
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    if save_model_b == 1:
         
        json_string = net.to_json()
        open(rootOutput + modelName, 'w').write(json_string)
        net.save_weights(rootOutput + modelName + '.h5',overwrite=True)
        
        
        
    #Write run-data to txt-file for documentation of the run:
    runDataFileName = rootOutput + modelName + '_runData.txt'
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
        s += "Letters are encoded as integers (1-4)" + "\n"
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
     
#    elif genSamplesFromRandomGenome_b > 0.5:
#        
#        s = "Samples from random genome, all generated with python code." + "\n"
#        s += "Genome data in file: " + randomGenomeFileName
    
    else: #fetch the data from an appropriate source

        s = "Pre-generated samples (ie not generated with the python code.)" + "\n"
        s += "Training samples from: " + fname  + "\n"
        s += "Validation samples from: " + vname  + "\n"
        s += "Test samples from: " + tname  + "\n"
        s += "shuffle_b = " + str(shuffle_b) + "\n"
        s += "inner_b = " + str(inner_b) + "\n"
        s += "sh uffleLength = " + str(shuffleLength) +  "\n"
        
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
    s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
    s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
    s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
    s += 'batchSize: ' + str(batchSize) + "\n"
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

    
    runDataFile.write(s)
    
    runDataFile.close()
            
    
    

#######################################################################################
    
########### FINE
    
#######################################################################################

'''
TODO: Make sure all arguments are properly handled by the existing code
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an LSTM on DNA sequences")
    ####################################################
    #Data
    ####################################################
    parser.add_argument("--path", type=str, default = '.')
    parser.add_argument("--rootGenome", type=str, default="data/")
    parser.add_argument("--fileName", type=str, default="hg19.fa")
    parser.add_argument("--on_binf_b", type=int, default = 1)
    parser.add_argument("--avoidChromo", nargs="+", default = 'chrX chrY chrM chr15 chr22') 
    ####################################################
    #Set up training schedule, model and run:
    ####################################################
    parser.add_argument("--nrOuterLoops", type=int, default=1)
    parser.add_argument("--firstIterNr", type=int, default=0)
    parser.add_argument("--nrOfRepeats", type=int, default=5)
    parser.add_argument("--firstRepeatNr", type=int, default=0)
    parser.add_argument("--testDataIntervalIdTotrainDataInterval_b", type=int, default=1)
    parser.add_argument("--nrEpochs", type=int, default=2)
    parser.add_argument("--batchSize", type=int, default=500)
    parser.add_argument("--stepsPerEpoch", type=int, default=100)
    parser.add_argument("--trainDataIntervalStepSize", type=str, default='2000000')
    parser.add_argument("--trainDataInterval", nargs="+", type=str, default='00000000 10000000')
    parser.add_argument("--nrTestSamples", type=int, default=500000)
    parser.add_argument("--testDataInterval", nargs="+", default='10000000 12000000')
    ####################################################
    #Modelling spec's
    ####################################################
    parser.add_argument("--exonicInfoBinaryFileName", type=str, default = '')
    parser.add_argument("--customFlankSize", type=int, default = 50)
    parser.add_argument("--overlap", type=int, default = 0)
    parser.add_argument("--pool_b", type=int, default = 0)
    parser.add_argument("--poolAt", nargs="+", default = '1 3')
    parser.add_argument("--maxPooling_b", type=int, default = 0)
    parser.add_argument("--poolStrides", type=int, default = 1)
    parser.add_argument("--lengthWindows", type=int, default = 4)
    parser.add_argument("--nrFilters", type=int, default = 256) 
    parser.add_argument("--filterStride", type=int, default = 1)
    parser.add_argument("--convLayers_b", type=int, default = 1)
    ####################################################
    #LSTMs:
    ####################################################
    parser.add_argument("--nrOfParallelLSTMstacks", type=int, default = 1)
    parser.add_argument("--nrLSTMlayers", type=int, default = 2)
    parser.add_argument("--tryAveraging_b", type=int, default = 1)
    parser.add_argument("--padding", type=str, default = 'valid')
    parser.add_argument("--labelsCodetype", type=int, default = 1)  #1: base pair prediction
    ####################################################
    #Final dense layers:
    ####################################################
    parser.add_argument("--finalDenseLayers_b", type=int, default = 1)
    parser.add_argument("--nrDenseLayers", type=int, default = 1)
    parser.add_argument("--hiddenUnits", nargs="+", default = "20")
    ####################################################
    #set-up
    ####################################################
    parser.add_argument("--genSamples_b", type=int, default=1)
    parser.add_argument("--shuffle_b", type=int, default=0)
    parser.add_argument("--dynSamplesTransformStyle_b", type=int, default = 0)
    parser.add_argument("--inclFrqModel_b", type=int, default = 0)
    parser.add_argument("--insertFrqModel_b", type=int, default = 0)
    parser.add_argument("--rootFrq", type=str, default = '/isdata/ßroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/')
    parser.add_argument("--frqFilename", type=str, default = "frqModel_k5.txt")
    parser.add_argument("--flankSizeFrqModel", type=int, default = 5)
    parser.add_argument("--exclFrqModelFlanks_b", type=int, default = 0)
    parser.add_argument("--augmentWithRevComplementary_b", type=int, default = 0)
    parser.add_argument("--dropout_b", type=int, default = 0)
    parser.add_argument("--dropoutVal", type=float, default = 0.0)
    parser.add_argument("--optimizer", type=str, default = 'ADAM')
    #default, but we use Adam here, so the value here isn't used
    parser.add_argument("--momentum", type=float, default = 0.1)
    parser.add_argument("--onlyOneRandomChromo_b", type=int, default = 0)
    parser.add_argument("--subStr", type=str, default = '_1Conv2LayerLstmLastAveraged_flanks50_win4_stride1_overlap0_dropout00')
    parser.add_argument("--learningRate", type=float, default = 0.001)
    args = parser.parse_args()
    fileGenome = args.rootGenome + args.fileName
    frqModelFileName = args.rootFrq + args.frqFilename
    modelName = 'models/modelLSTM_' + args.subStr
    modelDescr = args.subStr

    # Let's try to correctly parse this list of hidden unit sizes in the FC
    print(args.hiddenUnits)
    if type(args.hiddenUnits) is list:
        if len(args.hiddenUnits) == 1:
            hiddenUnits = [int(args.hiddenUnits)]
        else: 
            hiddenUnits = [int(x) for x in args.hiddenUnits]
    elif type(args.hiddenUnits) is str:
        # This is the default value
        hiddenUnits = [int(x) for x in args.hiddenUnits.split()]
    print(hiddenUnits)


    allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = args.labelsCodetype, 
                                          convLayers_b = args.convLayers_b, 
                                          nrLSTMlayers = args.nrLSTMlayers, 
                                          finalDenseLayers_b = args.finalDenseLayers_b,
                                          overlap = args.overlap, 
                                          learningRate = args.learningRate, 
                                          momentum = args.momentum,
                                          genomeFileName = fileGenome, 
                                          customFlankSize = args.customFlankSize, 
                                          inclFrqModel_b = args.inclFrqModel_b, 
                                          insertFrqModel_b = args.insertFrqModel_b, 
                                          exclFrqModelFlanks_b = args.exclFrqModelFlanks_b, 
                                          frqModelFileName = frqModelFileName, 
                                          flankSizeFrqModel = args.flankSizeFrqModel, 
                                          modelName = modelName, 
                                          trainDataIntervalStepSize = args.trainDataIntervalStepSize, 
                                          trainDataInterval0 = args.trainDataInterval.split(), 
                                          nrTestSamples = args.nrTestSamples, 
                                          testDataInterval = args.testDataInterval,
                                          genSamples_b = args.genSamples_b,
                                          lengthWindows = [args.lengthWindows],
                                          hiddenUnits = hiddenUnits,
                                          nrFilters = [args.nrFilters],
                                          padding = args.padding, 
                                          pool_b = args.pool_b, 
                                          maxPooling_b = args.maxPooling_b, 
                                          poolAt = args.poolAt, 
                                          poolStrides = args.poolStrides, 
                                          optimizer = args.optimizer, 
                                          dropoutVal = args.dropoutVal, 
                                          dropout_b = args.dropout_b, 
                                          augmentWithRevComplementary_b = args.augmentWithRevComplementary_b, 
                                          batchSize = args.batchSize, 
                                          nrEpochs = args.nrEpochs, 
                                          stepsPerEpoch = args.stepsPerEpoch, 
                                          shuffle_b = args.shuffle_b, 
                                          on_binf_b = args.on_binf_b, 
                                          path = args.path, 
                                          testDataIntervalIdTotrainDataInterval_b = args.testDataIntervalIdTotrainDataInterval_b
                                          )

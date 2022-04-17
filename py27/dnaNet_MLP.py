# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk
"""

'''

The code below -- all in a 'research state' -- was used for defining and training the pure convolutional networks reported in 
the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 


Notes:
   -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
    -- first version extracted from dnaNet_v7
    -- contains only code for feed forward/MLP models
    -- contains a single 'wrapper' function, 
    -- only the model 'makeMLPmodel' was used in the paper (makeIdmodel only for sampling tests) 

    -- This version:
    * allInOne_MLPmodel removed
    * allInOneWithDynSampling_MLPmodel now build as dnaNet_LSTM/allInOneWithDynSampling_ConvLSTMmodel
    * only generator part is here aimed for the MLP
   
   
##################################################################################################
# Usage:
##################################################################################################

The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.
    
In general: all functions called "allInOne"-something include/call all what is needed for training/validation fo a particular model.
So allInOneWithDynSampling_MLPmodel, will train/test a MLP model; the function calls the code for building the model, for compiling 
it and for dynamically sampling from the desired data. The parameters of the function allow to specify the model, the sampling and 
the training.   
READ the Usage section of the dnaNet_LSTM module for some information on how the training
is structured (text section beloow 'Usage' and until 'Import module').


####################################################

Import module:

####################################################

import dnaNet_MLP as dnaNet
   

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


rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
fileName = r"hg38_chr22.txt"
fileGenome = rootGenome +fileName


#whole hg38 seq here:
rootGenome = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/"
fileName = r"GCF_000001405.38_GRCh38.p12_genomic_filter.fna"
fileGenome = rootGenome +fileName

 
####################################################

#Set up training schedule, model and run:

####################################################


#For looking at the flow:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 0 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 1
batchSize = 10
stepsPerEpoch = 10
trainDataIntervalStepSize = 200
trainDataInterval = [20000,21000]
nrTestSamples = 10
testDataInterval = [21000,22000]


#For testing a set-up (check that it runs, how long training time is ...):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 5
firstRepeatNr = 0
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 10
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 20000
trainDataInterval = [200000, 23000000]
nrTestSamples = 5000
testDataInterval = [30000000,31000000]

#In anger:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 200
firstRepeatNr = 0 #loads in model from repeatNr [this number] - 1...!
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]



#Modelling spec's
exonicInfoBinaryFileName  = ''
inclFrqModel_b = 0
insertFrqModel_b = 0
customFlankSize = 50
nrHiddenUnits = [150, 150, 150]


#set-up
dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
file = "frqModel_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
exclFrqModelFlanks_b = 0
#!!!:
augmentWithRevComplementary_b = 0
##
dropout_b = 0
dropoutVal = 0.15
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.1 #default, but we use Adam here, so the value here isn't used
#learningRate = learningRate
chromoNameBound = 100
onlyOneRandomChromo_b = 0
avoidChromo = [] #['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1 
 


#subStr = '3MLPLayer_150_150_150_flanks50'
subStr = 'testSampling'
 
learningRate = 0.001
modelName = 'modelMLP_' + subStr
modelDescr = subStr


rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/MLP3/trainTestSplit_80_20/"

labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeMLPmodel'
dnaNet.allInOneWithDynSampling_MLPmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr,  learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  dropoutVal = dropoutVal, nrHiddenUnits = nrHiddenUnits, optimizer = optimizer,  augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#########################
#For test of sampling:
subStr = 'testSampling'

learningRate = 0.001
modelName = 'modelMLP_' + subStr
modelDescr = subStr

rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/MLP/testSampling/"

labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeIDmodel'

testOfSamplingOnly_b = 1
firstRepeatNr = 0
samplingCountsTotal_train, samplingCountsTotal_test = dnaNet.allInOneWithDynSampling_MLPmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr,  learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  dropoutVal = dropoutVal, nrHiddenUnits = nrHiddenUnits, optimizer = optimizer,  augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b, testOfSamplingOnly_b = testOfSamplingOnly_b) 

loadfile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/MLP/testSampling/testOfSampling_MLP_samplingCountsTotal_train_repeatNr190.p'
t= pickle.load(open(loadfile,"rb"))

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

import dnaNet_dataGen as dataGen #all smpling aso is here

import cPickle as pickle

from scipy.sparse import csr_matrix

#import graphviz
#import pydot


##############################################################
## Identity model: only for sampling testing
##############################################################

#def makeIDmodel(inputLength, encodingShape = 1):
def makeIDmodel(inputLength, batchSize):
    '''
    Identity: outputs the input.
    '''

    print('Build ID model...')

    inputs = Input(shape=(inputLength), batch_shape = (batchSize, inputLength))
 
    
    output = Lambda(lambda x: backend.identity(x))(inputs)
    
#    output = backend.int(output)

    model = Model(inputs= [inputs], outputs=output)

    print("... build model.")
        
    return model
    
    

############################## 
##        MLP's
##############################

def makeMLPmodel(inputDim, 
                    nrHiddenUnits = [100], 
                    outputSize = 4,
                    dropoutVal = 0.25):
    '''
    Feed-forward network model.
    
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    nrHiddenUnits: list giving the nr of units for each consecutive hidden layer (the length of the list 
    therefore gives the nr of layers).
    outputSize = the size of the output layer, here 4
    '''

    print('Build MLP model...')  
    
    model = Sequential()
    
#    model.add(Dropout(dropoutVal, noise_shape=None, seed=None))  
    model.add(Dense(nrHiddenUnits[0], input_dim=inputDim, init='uniform', activation='relu')) 
    model.add(Dropout(dropoutVal, noise_shape=None, seed=None))  
    for i in range(1, len(nrHiddenUnits)):    
        
        model.add(Dense(nrHiddenUnits[i], init='uniform', activation='relu'))
        model.add(Dropout(dropoutVal, noise_shape=None, seed=None)) 


    model.add(Dense(outputSize, kernel_initializer='uniform', activation='softmax'))  #changed from sigmiod to softmax 
    
    return model


          
def allInOneWithDynSampling_MLPmodel(rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/",
                                          nrOuterLoops = 1,
                                          firstIterNr = 0,
                                          nrOfRepeats = 1,
                                          firstRepeatNr = 0,
                                          loss = "categorical_crossentropy", 
                                          usedThisModel = 'makeMLPmodel', #set this manually if restarting
                                          onHecaton_b = 0,
            learningRate = 0.01,
            momentum = 1.0,
            trainTestSplitRatio = 0.8,
            trainDataIntervalStepSize = 100000, 
            trainDataInterval0 = [0,200000] ,
            testDataIntervalIdTotrainDataInterval_b = 0,     
            nrTestSamples = 20000,
            testDataInterval = [400000, 600000], 
            customFlankSize_b = 1, 
            customFlankSize = 50,
            genSamples_b = 0, 
            genomeFileName = '',
            chromoNameBound = 1000,
            exonicInfoBinaryFileName  = '',
            outputEncodedOneHot_b = 1,
            labelsCodetype = 0,
            outputEncodedInt_b = 0,
            onlyOneRandomChromo_b = 0,
            avoidChromo = [],
            genSamplesFromRandomGenome_b = 0, 
            randomGenomeSize = 4500000, 
            randomGenomeFileName = 'rndGenome.txt',
            getOnlyRepeats_b = 0,
            augmentWithRevComplementary_b = 0,
            augmentTestDataWithRevComplementary_b = 0,
            inclFrqModel_b = 0,
            frqModelFileName = '',
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0, 
            optimizer = 'ADAM',
            batchSize = 50, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=4,
            letterShape = 4, # size of the word
            dropoutVal = 0.25,
            nrHiddenUnits = [50],
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamplesDyn/Ecoli/model1', 
            modelDescription = 'MLP type ... to be filled in!',
            on_binf_b = 1, 
            testOnly_b = 0,
            testOfSamplingOnly_b = 0):
 
    '''     labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding and not including the frq model 
                (inclFrqModel_b = 0).
    '''
                
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"

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
                    genomeArray, repeatArray, exonicArray, genomeString = dataGen.encodeGenome(fileName = genomeFileName, chromoNameBound = chromoNameBound, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0)
                    lGenome = len(genomeArray)
                    genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
                elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                    genomeDictArray, repeatInfoDictArray, exonicInfoDictArray, genomeDictString = dataGen.encodeGenome(fileName = genomeFileName, chromoNameBound = chromoNameBound, exonicInfoBinaryFileName  = exonicInfoBinaryFileName ,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1)
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
            
            if testOfSamplingOnly_b == 1:
                
                samplingCountsThisRepeat_train = np.zeros(shape = lGenome, dtype = 'int32')
                samplingCountsTotal_train = np.zeros(shape = lGenome, dtype = 'int32')

                samplingCountsThisRepeat_test = np.zeros(shape = lGenome, dtype = 'int32')
                samplingCountsTotal_test = np.zeros(shape = lGenome, dtype = 'int32')

                    
            if inclFrqModel_b == 1: 
                
                sizeInputConv = int((2*(customFlankSize-exclFrqModelFlanks_b*flankSizeFrqModel)+1))
                sizeInputMlp = int((2*(customFlankSize-exclFrqModelFlanks_b*flankSizeFrqModel)+1)*letterShape)
            
                XfrqConv = np.zeros(shape = (batchSize, 1, letterShape))
            
            else:
                
                sizeInputConv = 2*customFlankSize
                sizeInputMlp = int(2*customFlankSize*letterShape)
                
            if augmentWithRevComplementary_b == 0:
                Xconv = np.zeros(shape = (batchSize, sizeInputConv, letterShape))
            else:
                Xconv = np.zeros(shape = (2*batchSize, sizeInputConv,letterShape))

    
            print("sizeInput is set to: ", sizeInputMlp)
            
                  
            #we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so; the data for testing is loaded after
            #the training is done (below) so as to avoid spending the memory needed for the test data during 
            #the training part: 
            frqModelDict = {}
            if inclFrqModel_b == 1:
                
                frqModelDict = frqM.getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)
 
            
            #Dynamically fetch small sample batches; this runs in an infinite loop
            #in parallel with the fit_generator call below (and stops when that is done)
            #For the purpose of separating train and test data we pass in a boolean array indicating
            #the indices to sample; we generate this array based on the set trainTestSplitRatio (to 
            #save memory we use boolean arrays rather than just the set of indices; this implies an extra if-clause
            #in the sampling, but with a typical 80-20 split, this should be of small concern):
            trainSetIndicator = np.zeros(shape = lGenome, dtype = 'int8')
            testSetIndicator = np.zeros(shape = lGenome, dtype = 'int8')            
            for i in range(lGenome):
                
                #0-1 toss with prob = trainTestSplitRatio:
                ind = np.random.binomial(1, p= trainTestSplitRatio)
                trainSetIndicator[i] = ind
                testSetIndicator[i] = 1 - ind
        
            print "Train-test split sizes (nr of positions of genome array): ", np.sum(trainSetIndicator), np.sum(testSetIndicator) 

##            this is just cosmetic/for convenience -- eases the call to the generator in fit_generator below: 
#            def myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b):
  
#                dataGen.myGenerator_MLP(customFlankSize = customFlankSize,
#                                batchSize = batchSize, 
#                                inclFrqModel_b = inclFrqModel_b, 
#                                labelsCodetype = labelsCodetype, 
#                                forTrain_b = forTrain_b,                 
#                                genomeArray = genomeArray, 
#                                repeatArray = repeatArray,
#                                exonicArray = exonicArray,
#                                trainSetIndicator = trainSetIndicator,
#                                testSetIndicator = testSetIndicator,
#                                getOnlyRepeats_b = getOnlyRepeats_b,
#                                genomeString = genomeString,
#                                frqModelDict = frqModelDict,
#                                flankSizeFrqModel = flankSizeFrqModel,
#                                exclFrqModelFlanks_b = exclFrqModelFlanks_b,
#                                outputEncodedOneHot_b = outputEncodedOneHot_b,
#                                outputEncodedInt_b = outputEncodedInt_b,
#                                shuffle_b = shuffle_b,
#                                inner_b = inner_b,
#                                shuffleLength = shuffleLength, 
#                                augmentWithRevComplementary_b = augmentWithRevComplementary_b,  
#                                Xconv = Xconv,
#                                XfrqConv = XfrqConv
#                                )

            def myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b):
                
                if testOfSamplingOnly_b == 0: #standard use case
                    while 1:
                        
                        if forTrain_b == 1: #for train set
                            X,Y = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = trainSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                        else: #for test set        
                            X,Y = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = testSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
    
                        if inclFrqModel_b == 1:
                
                            Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                            Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                            XfrqConv[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
                            
                            Xmlp  = Xconv.reshape((Xconv.shape[0],Xconv.shape[1]*Xconv.shape[2]))
                            Xfrq =  Xconv.reshape((XfrqConv.shape[0],XfrqConv.shape[1]*XfrqConv.shape[2]))
                            
                            yield([Xfrq, Xmlp],Y)
                    
                        
                        elif inclFrqModel_b == 0:
                            
                            Xmlp = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    
                            
                            yield(Xmlp,Y)
                            
                elif testOfSamplingOnly_b == 1: #for testing the sampling
                    
                    while 1:
                        
                        if forTrain_b == 1: #for train set and used for training
                            Z, Z = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = trainSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, testSampling_b = testOfSamplingOnly_b)
                            yield(Z,Z)
                        
                        elif forTrain_b == -1: #for train set but used for testing sampling in training part (the two are split)
                            Z, Z = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = trainSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, testSampling_b = testOfSamplingOnly_b)
                            
                            yield(Z) #in contrast to forTrain_b = 1
                        
                        else: #for test set; used for testing sampling in testing part (the two are split)
                            Z, Z = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = testSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, testSampling_b = testOfSamplingOnly_b)
                            yield(Z)
                        

        #This first part is only fo testing the sampling; after this section the code
        #is for the training and testing of the model and that only      
        if testOfSamplingOnly_b == 1: #for testing the sampling
        
            #Record the settings for the sampling test:
            testFileName = rootOutput + 'bigLoopIter' + str(n)
            
            #Write run-data to txt-file for documentation of the run:
            runDataFileName = testFileName + '_runData_samplingTestGeneratorMLP.txt'
            runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
            
            s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
            s += "Test of sampling generator for MLP\n" 
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
            #Which genome data were used?:
            if genSamples_b > 0.5:
                s = "Samples generated with python code from real genome." + "\n"
                s += "Genome data in file: " + genomeFileName + "\n"
                s += "exonicInfoBinaryFileName: " + exonicInfoBinaryFileName + "\n"
                s += 'inclFrqModel_b: ' + str(inclFrqModel_b) + "\n"
                s += 'frqModelFileName: ' + frqModelFileName + "\n"  
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
                s += "trainTestSplitRatio:" + str(trainTestSplitRatio)  + "\n"
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
            s += "chromoNameBound: " + str(chromoNameBound) + "\n"
            s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
            s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
            s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n"
            s += "avoidChromo: " + str(avoidChromo)  + "\n" 
            s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
            s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
            s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
            s += 'batchSize: ' + str(batchSize) + "\n"
            s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
            s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
            s += 'letterShape: ' + str(letterShape) + "\n" 
            s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
        
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
        
            runDataFile.close()
            #Write run-data to txt-file for documentation of this SAMPLING TEST run: DONE
            
            #dump the train-test split -- as sparse!:         
            dumpFile = rootOutput + r'/testOfSampling_LSTM_trainSetIndicator.p'
            trainSetIndicatorToDump = csr_matrix(trainSetIndicator)
            pickle.dump( trainSetIndicatorToDump, open(dumpFile, "wb") )
            dumpFile = rootOutput + r'/testOfSampling_LSTM_testSetIndicator.p'
            testSetIndicatorToDump = csr_matrix(testSetIndicator)
            pickle.dump( testSetIndicatorToDump, open(dumpFile, "wb") )
            
            #Run the sampling and record the result: for the sake of running this in on
            #the gpu, this is done by predicting the input (which is batches of indices of 
            #sampled positions) with the identity model:
            idNet = makeIDmodel(inputLength = 1, batchSize = batchSize) #, encodingShape = 1) 
            idNet.compile(loss='mean_absolute_error', optimizer='ADAM', metrics=['accuracy'])    
            forTrain_b = 1
            history = idNet.fit_generator(myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
            #Now do the sampling; we call the sampling through each repeat (as in training) and test the sampling after each repeat (as in testing)  
            for k in range(firstRepeatNr, nrOfRepeats):
                
                print "Sampling testing, now at repeat ", k 
                
#                #read in history if start at repeat nr > 0:
#                if k == firstRepeatNr and k > 0:
#                    loadfile = rootOutput + r'/testOfSampling_MLP_samplingCountsTotal_train_repeatNr' + str(k-1) + '.p'
#                    samplingCountsTotal_train = pickle.load(open(loadfile,"rb"))
#                    loadfile = rootOutput + r'/testOfSampling_MLP_samplingCountsTotal_test_repeatNr' + str(k-1) + '.p'
#                    samplingCountsTotal_test = pickle.load(open(loadfile,"rb"))
                    
                
                #First part: sampling as in training
                samplingCountsThisRepeat_train = 0*samplingCountsThisRepeat_train #reset
                
                
                for n in range(nrEpochs):
                    print "epoch ", n
#                    for s in range(stepsPerEpoch):
#                        print "step ", s
                        
#                        i = 0
#                        for (x,y,z) in myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b):
#
#                            i += 1
#                            if i == batchSize:
#                                break 
#                    pred = idNet.predict(myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b), batch_size = batchSize)
                    gen = myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b = -1)
                    pred = idNet.predict_generator(gen, steps= stepsPerEpoch)
                    
                    pred = pred.flatten()
                    pred = pred.astype(np.int64, casting='unsafe', copy=False) #MUST be int64!
                    
#                    print pred
                    
                    for u in pred:

                        samplingCountsThisRepeat_train[u] += 1
                
                samplingCountsTotal_train += samplingCountsThisRepeat_train
                
                
                #2nd part: Now get the sampling for the test part:
                samplingCountsThisRepeat_test = 0*samplingCountsThisRepeat_test #reset
                
                gen = myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b = 0)
                pred = idNet.predict_generator(gen, steps = np.int(float(nrTestSamples)/batchSize))
                    
                pred = pred.flatten()
                pred = pred.astype(np.int64, casting='unsafe', copy=False) #MUST be int64!
                
#                    print pred
                
                for u in pred:

                    samplingCountsThisRepeat_test[u] += 1
                
                samplingCountsTotal_test += samplingCountsThisRepeat_test                
                
                
                #dump result for selected repeats; we take though only those accumalted since last dump; and dump the results as sparse:
                if k%10 == 0:                
                
                    samplingCountsTotal_train_toDump = csr_matrix(samplingCountsTotal_train)       
                    dumpFile = rootOutput + r'/testOfSampling_LSTM_samplingCountsTotal_train_repeatNr' + str(k) + '.p'
                    pickle.dump( samplingCountsTotal_train_toDump, open(dumpFile, "wb") )
                    #reset
                    samplingCountsTotal_train = 0*samplingCountsTotal_train
                    
                    samplingCountsTotal_test_toDump = csr_matrix(samplingCountsTotal_test)
                    dumpFile = rootOutput + r'/testOfSampling_LSTM_samplingCountsTotal_test_repeatNr' + str(k) + '.p'
                    pickle.dump( samplingCountsTotal_test_toDump, open(dumpFile, "wb") )
                    #reset
                    samplingCountsTotal_test = 0*samplingCountsTotal_test
                
                
            #the code below should not be run when testing only sampling: 
            
            return samplingCountsTotal_train, samplingCountsTotal_test #continue
           

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
        #From here code is for the training and testing of the model and that only 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
                    
        #if augmentWithRevComplementary_b == 1 the generated batches contain 2*batchSize samples:
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
    
    
        if testOnly_b == 0: #just means that we're running a regular traning/testing session
    
            #Write run-data to txt-file for documentation of the run:
            runDataFileName = modelFileName + '_runData.txt'
            runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
            
            s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
            s += modelDescription  + "\n"  
#            s += 'ExonRepaetOther (ERO) prediction model included?: ' + str(fusedWitEROmodel_b) + "\n"  
#            s += 'eroModelFileName: ' + eroModelFileName + "\n" 
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
                s += 'inclFrqModel_b: ' + str(inclFrqModel_b) + "\n"
                s += 'frqModelFileName: ' + frqModelFileName + "\n"  
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
                s += "trainTestSplitRatio:" + str(trainTestSplitRatio)  + "\n"
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
            s += "chromoNameBound: " + str(chromoNameBound) + "\n"
            s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
            s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
            s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n"
            s += "avoidChromo: " + str(avoidChromo)  + "\n" 
            s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
            s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
            s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
            s += 'learningRate: ' + str(learningRate) + "\n"
            s += 'batchSize: ' + str(batchSize) + "\n"
#            s += 'dropout_b: ' + str(dropout_b) + "\n"
            s += 'dropoutVal: ' + str(dropoutVal) + "\n"
#            s += 'tryAveraging_b: ' + str(tryAveraging_b) + "\n"
#            s += 'pool_b: ' +  str(pool_b) + "\n"
#            s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
#            s += 'poolAt: ' +  str(poolAt) + "\n"
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
            s = 'nrHiddenUnits: ' + str(nrHiddenUnits)  + "\n" 
        
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
        
            runDataFile.close()
            #Write run-data to txt-file for documentation of the run: DONE


        #Run series of repeated training-and-testing sessions each consisting in nrEpochs rounds:
        for k in range(firstRepeatNr, nrOfRepeats):       
            
            #in first outer-iteration build the model; thereafter reload the latest stored version (saved below)
            if n == 0 and k == 0 and testOnly_b == 0: 
        
                if inclFrqModel_b == 0:
            
                    net = makeMLPmodel(inputDim = sizeInputMlp, nrHiddenUnits = nrHiddenUnits, outputSize = sizeOutput)
 
                    usedThisModel = 'makeMLPmodel'
                                                 
                        
                elif inclFrqModel_b == 1:
                    
                    print "TO BE DONE: fuse frq-model with  MLP. Returns here!"
                    return
                 
                    
                #Call a summary of the model:
                net.summary()
                #Save a plot of the model:
                if onHecaton_b == 0:
                    plot_model(net, to_file= rootOutput + modelName + '_plot.png', show_shapes=True, show_layer_names=True)
                
                    
            else:
    
                #reload the model from previous iter/repeat
                if k == 0 and testOnly_b == 0:
                    modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n-1) + '_repeatNr' + str(nrOfRepeats-1)
                elif testOnly_b == 0:
                    modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n) + '_repeatNr' + str(k-1)
                elif testOnly_b == 1: #we load the model for this repearNr and run a test on it:
                    modelFileNamePrevious = rootOutput + modelName + '_bigLoopIter' + str(n) + '_repeatNr' + str(k)
                    
                net = model_from_json(open(modelFileNamePrevious).read())
                net.load_weights(modelFileNamePrevious +'.h5')
        
                print("I've now reloaded the model from the previous iteration (for test-only: for this repeatNr: ", modelFileNamePrevious)
                        

        
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

                forTrain_b = 1
                history = net.fit_generator(myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
           
    
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
    
    
                      
            #test it. First read in the test data:
            #If so desired, we fetch the output from the frq model if we want to include it in the training and testing; 
            #the test set shall also include the frq model output if so: 
            print "Now testing ..."
            if testDataIntervalIdTotrainDataInterval_b == 0:
               
                if inclFrqModel_b == 1:
                        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                            flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)

                        
                    #Split the test data as the training data:
                    nrOfTestSamples = Xt.shape[0]
                    Xconv_t = np.zeros(shape = (nrOfTestSamples, sizeInputConv, letterShape))
                    XfrqConv_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
        
                    Xconv_t[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                    Xconv_t[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                    XfrqConv_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
                                             
                    Xmlp_t  = Xconv_t.reshape((Xconv_t.shape[0],Xconv_t.shape[1]*Xconv_t.shape[2]))
                    Xfrq_t =  Xconv.reshape((XfrqConv_t.shape[0],XfrqConv_t.shape[1]*XfrqConv_t.shape[2]))
         
                    
                else:
        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                              flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  
                                                              onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
                
                    Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]))
                    
                if inclFrqModel_b == 1:
                    
                    score, acc = net.evaluate([Xfrq_t, Xmlp_t], Yt, batch_size=batchSizeReal, verbose=1)

                else:
                    
                    score, acc = net.evaluate(Xt,Yt, batch_size=batchSizeReal, verbose=1)
                    
                    
            elif testDataIntervalIdTotrainDataInterval_b == 1: #we test using the dynamic sampling
                                
                print "In test: Test data interval id to train data interval!"
                print "But: train-test split ratio set to: %f" % trainTestSplitRatio
                forTrain_b = 0
                score, acc = net.evaluate_generator(myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b), steps = np.int(float(nrTestSamples)/batchSize))
                    

                
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
            #load the history if restarting at repeatNr > 0:
            if k == firstRepeatNr and firstRepeatNr > 0:
                testHistoryTotal['acc'] = pickle.load( open( modelFileName + '_repeatNr' + str(k-1) + '_testing_acc_vs_epoch.p', "rb" ) )
                testHistoryTotal['loss'] = pickle.load( open( modelFileName + '_repeatNr' + str(k-1) + '_testing_loss_vs_epoch.p', "rb" ) )
                
            testHistoryTotal['acc'].append(acc)
            testHistoryTotal['loss'].append(score)
            
            #dump the info:
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_testing_acc_vs_epoch.p'
            pickle.dump( testHistoryTotal['acc'], open(dumpFile, "wb") )
            dumpFile = modelFileName + '_repeatNr' + str(k) + '_testing_loss_vs_epoch.p'
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
   



#######################################################################################
    
########### FINE
    
#######################################################################################


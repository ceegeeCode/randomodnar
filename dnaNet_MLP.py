# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk
"""

'''
Notes:
    -- first version extracted from dnaNet_v7
    -- contains only code for MLP models
    --OBS: not sure that the code runs as is (been a long time since I ran it). So undoubtedly needs some nursing ....
   
Usage:
    
In general: all functions called "allInOne"-something include/call all what is needed for training/validation fo a particular model.
So allInOneWithDynSampling_MLPmodel, will train/test a MLP model; the function calls the code for building the model, for compiling 
it and for dynamically sampling from the desired data. The parameters of the function allow to specify the model, the sampling and 
the training.
    

(will add more)     

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
##        MLP's
##############################

def makeMLPmodel(inputDim, 
                    nrHiddenUnits = [100], 
                    outputSize = 4,
                    dropOut = 0.0):
    '''
    network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    nrHiddenUnits: list giveing the nr of units for each consecutive hidden layer (the length of the list 
    therefore gives the nr of layers).
    outputSize = the size of the output layer, here 4
    '''

    print('Build MLP model...')  
    
    model = Sequential()
    
    model.add(Dense(nrHiddenUnits[0], input_dim=inputDim, init='uniform', activation='relu')) 
    model.add(Dropout(0.1, noise_shape=None, seed=None))  
    for i in range(1, len(nrHiddenUnits)):    
        
        model.add(Dense(nrHiddenUnits[i], init='uniform', activation='relu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None)) 


    model.add(Dense(outputSize, init='uniform', activation='softmax'))  #changed from sigmiod to softmax 
    
    return model



#All in one run
    #labelsCodetype NOT ENCORPORATED
def allInOne_MLPmodel(loss = "categorical_crossentropy", 
                      optimizer = "ADAM",
            learningRate = 0.01,
            momentum = 1.0,
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
            batchSize = 50, 
            nrEpochs = 100,
            sizeOutput=4,
            letterShape = 4, # size of the word
            nrHiddenUnits = [50],
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamples/CElegans/model3', 
            modelDescription = 'MLP type ... to be filled in!',
            on_binf_b = 1):
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"D:/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_nets/"

    
    if genSamples_b > 0.5: #generate a set of random data acc to the sizes set

        #if a genomeFileName is specified, use that genome:
        if len(genomeFileName) > 0:
            fromGenome_b = 1
        else:
            fromGenome_b = 0
    
        X,Y, genomeSeqSourceTrain = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTrainSamples, startAtPosition = trainDataInterval[0], endAtPosition = trainDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        sizeInput = X.shape[1]
        print("X shape", X.shape)

        avoidChromo.append(genomeSeqSourceTrain) #to avoid getting val data from the same chromo as the training data 
        Xv,Yv, genomeSeqSourceVal = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrValSamples, startAtPosition = valDataInterval[0], endAtPosition = valDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        
        avoidChromo.append(genomeSeqSourceVal) ##to avoid getting test data from the same chromo as the training and validation data 
        Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        

    elif genSamplesFromRandomGenome_b > 0.5: #generate a set of random data acc to the sizes set

        #generate random genome of set size:   
        genRandomGenome(length = randomGenomeSize, fileName = root + randomGenomeFileName, on_binf_b = on_binf_b) #will write the generated genome sequence to the file  

        X,Y = genSamples_I(fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  flankSize = customFlankSize, nrSamples = nrTrainSamples, startAtPosition = trainDataInterval[0], endAtPosition = trainDataInterval[1], augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        sizeInput = X.shape[1]

        Xv,Yv = genSamples_I(fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  flankSize = customFlankSize, nrSamples = nrValSamples, startAtPosition = valDataInterval[0], augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)

        Xt,Yt = genSamples_I(fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  flankSize = customFlankSize,  nrSamples = nrTrainSamples, startAtPosition = testDataInterval[0], augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        

    else: #fetch the data from an appropriate source

        #Using the getData2-fct to fetch data:  
        fname=root + r"training.dat"
        vname = root + r"validation.dat"
        tname=root + r"test.dat"
    
        
        X,Y = getData2(fname, letterShape, sizeOutput,  loadRecsInterval = trainDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
        sizeInput = X.shape[1]
                
        Xv,Yv = getData2(vname, letterShape, sizeOutput,  loadRecsInterval = valDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)      
            
        Xt,Yt = getData2(tname, letterShape, sizeOutput,  loadRecsInterval = testDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
    
    
    batch_size = min(batchSize,max(1,len(X)/20))
    #print batch_size, nrEpoch
    
    inputDim = int(sizeInput*letterShape)

    net = makeMLPmodel(inputDim = inputDim, nrHiddenUnits = nrHiddenUnits, outputSize = sizeOutput)


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

#    net.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    
    #net.compile(loss='categorical_crossentropy', optimizer='Adam')
    #net.compile(loss='binary_crossentropy', optimizer=sgd)
    #xor.compile(loss="hinge", optimizer=sgd)
    #xor.compile(loss="binary_crossentropy", optimizer=sgd)
      
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    Xv = Xv.reshape((Xv.shape[0],Xv.shape[1]*Xv.shape[2]))
    Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]))
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
    plt.savefig(rootOutput + modelName + '_training_validation_acc_vs_epoch.pdf')
#    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(rootOutput + modelName + '_training_validation_loss_vs_epoch.pdf')
#    plt.show()

    #test it:
    score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
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
    
    runDataFile.close()
        
        
        
    
#DYN SAMPLING TO BE UPDATED!!    
def allInOneWithDynSampling_MLPmodel(loss = "categorical_crossentropy", 
            learningRate = 0.01,
            momentum = 1.0,
            nrTrainSamples = 100000,
            trainDataInterval = [0,200000] ,  
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
            augmentTestDataWithRevComplementary_b = 0,
            inclFrqModel_b = 0,
            frqModelFileName = '',
            flankSizeFrqModel = 4,
            exclFrqModelFlanks_b = 0, 
            batchSize = 50, 
            nrEpochs = 100,
            stepsPerEpoch = 5, 
            sizeOutput=4,
            letterShape = 4, # size of the word
            nrHiddenUnits = [50],
            shuffle_b = 0, 
            inner_b = 1, 
            shuffleLength = 5,
            save_model_b = 1, 
            modelName = 'ownSamplesDyn/Ecoli/model1', 
            modelDescription = 'MLP type ... to be filled in!',
            on_binf_b = 1):
                
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"D:/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_nets/"

    
    if genSamples_b > 0.5: #generate a set of random samples from genome or random data acc to the input/the sizes set

        #if a genomeFileName is specified, use that genome:
        if len(genomeFileName) > 0:
            
            fromGenome_b = 1
            
            startAtPosition = trainDataInterval[0]
            endAtPosition = trainDataInterval[1]
            
            #read in the genome sequence:
            if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
                genomeArray, repeatInfoArray, genomeString = encodeGenome(genomeFileName, startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0)
                lGenome = len(genomeArray)
                genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
            elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                genomeDictArray, repeatInfoDictArray, genomeDictString = encodeGenome(genomeFileName,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1)
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

        #we fetch the output from the frq model if we want to include it in the training and testing; 
        #the test set shall also includes the frq model output if so: 
        frqModelDict = {}
        if inclFrqModel_b == 1:
            
            frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)

            #Read in the test data we avoid the chromos used for training:    
            avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
            Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, inclFrqModel_b = inclFrqModel_b,
                                                    flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
                                                    
        else:

            #Read in the test data we avoid the chromos used for training:    
            avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
            Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
                
        
        #Dynamically fetch small sample batches; this runs in an infinite loop
        #in parallel to the fit_generator call below (and stops when that is done)
        def myGenerator(customFlankSize,batchSize):
            while 1:
                X,Y = genSamplesForDynamicSampling(nrSamples = batchSize, genomeArray = genomeArray, inclFrqModel_b = inclFrqModel_b,
                                 genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, flankSize = customFlankSize, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b,   shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
#                sizeInput = X.shape[1]
                yield(X, Y)
    #            print "X shape", X.shape
        
    
    batch_size = batchSize
    #print batch_size, nrEpoch
    
    if inclFrqModel_b == 1: 
        inputDim = int((2*(customFlankSize-exclFrqModelFlanks_b*flankSizeFrqModel)+1)*letterShape)
    else:
        inputDim = int(2*customFlankSize*letterShape)

    net = makeMLPmodel(inputDim = inputDim, nrHiddenUnits = nrHiddenUnits, outputSize = sizeOutput)

#    utils.plot_model(net, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
    sgd = SGD(lr = learningRate, decay=1e-6, momentum= momentum, nesterov=True)
    #sgd = SGD(lr=0.01)
    
#    net.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    net.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    
    #net.compile(loss='categorical_crossentropy', optimizer='Adam')
    #net.compile(loss='binary_crossentropy', optimizer=sgd)
    #xor.compile(loss="hinge", optimizer=sgd)
    #xor.compile(loss="binary_crossentropy", optimizer=sgd)
      
    #net.fit(X, Y, batch_size=batch_size, epochs=nrEpochs, show_accuracy=True, verbose=0)
#    history = net.fit_generator(X, Y, batch_size=batch_size, epochs = nrEpochs, verbose=1, validation_data=(Xv,Yv) )

    history=net.fit_generator(myGenerator(customFlankSize,batchSize), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
   

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(rootOutput + modelName + '_training_acc_vs_epoch.pdf')
#    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(rootOutput + modelName + '_training_loss_vs_epoch.pdf')
#    plt.show()

    #test it:
    Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]))
    score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
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
        s = "Samples generated with python code from real genome." + "\n" + "Dynamic sampling for training (validation not needed)." + "\n" + "Fixed sample set for testing." + "\n"
        s += "Genome data in file: " + genomeFileName + "\n"
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
    s += 'learning rate: ' + str(learningRate)  + "\n" 
    s += 'momentum: ' + str(momentum) + "\n" 
    s += 'batchSize: ' + str(batchSize) + "\n"
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
    
    runDataFile.close()


#######################################################################################
    
########### FINE
    
#######################################################################################


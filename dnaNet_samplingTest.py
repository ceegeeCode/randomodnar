 # -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:47:48 2020

@author: Christian Grønbæk

"""

'''
The code below -- all in a 'research state' -- was used for analysis of th sampling used in the training of the neural networks 
as reported in the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 

What is found here is a small module for carrying out analysis of the sampling quality, ie that the sampling
done in training and testing is uniform accross the genome.

Notes:

    -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
    -- the examples below take for grated that the training process was also run in a 'test the sampling' mode (see eg the
        function allInOneWithDynSampling_ConvLSTMmodel in the module dnaNet_LSTM; find its code and follow the testOfSamplingOnly_b parameter).


##################################################################################################
# Usage:
##################################################################################################

The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.


import dnaNet_samplingTest as sampleT


#Step 1. Get hold of the genome string exactly as when generating the sampling-data:

import dnaNet_dataGen as dataGen

genomeFileName = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/GCF_000001405.38_GRCh38.p12_genomic_filter.fna" 

testDataIntervalIdTotrainDataInterval_b = 1
trainDataInterval = [0,3000000000]
                
startAtPosition = trainDataInterval[0]
endAtPosition = trainDataInterval[1]

exonicInfoBinaryFileName  = ''
outputEncodedOneHot_b = 1
labelsCodetype = 0
outputEncodedInt_b = 0
chromoNameBound = 100 #mouse:65
onlyOneRandomChromo_b = 0
avoidChromo = [] #['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1 

genomeArray, repeatArray, exonicArray, genomeString = dataGen.encodeGenome(fileName = genomeFileName, chromoNameBound = chromoNameBound, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0)
lGenome = len(genomeArray)


#Step 2:  extract the stats:
trainTestSplitRatio = 0.8

rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/"
lastRepeatNr = 200
epochs = 100
batchSize = 500
nrSteps = 100
nrTestSamples = 1000000
qualifiedPositions = qualifiedPositions
windowList = [100000, 1000000]
resultsDict = sampleT.getSamplingStats(rootOutput, lastRepeatNr, epochs, batchSize, nrSteps, nrTestSamples, qualifiedPositions, windowList) #, trainIndicatorSet, testIndicatorSet )


################################################################
## Chek that train/test indicator sets have the right proportions
################################################################

load the sampling data:
rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/"

import cPickle as pickle

loadFile = rootOutput + 'testOfSampling_LSTM_trainSetIndicator.p'
trainIndicatorSet = pickle.load(open( loadFile, "rb"))

loadFile = rootOutput + 'testOfSampling_LSTM_testSetIndicator.p'
testIndicatorSet = pickle.load(open( loadFile, "rb"))

#get the sizes and the ratio
lTrain = trainIndicatorSet.shape[1]
lTest = testIndicatorSet.shape[1]
ratio = float(lTest/lTrain)
print lTrain, lTest, ratio

'''

import numpy as np

import cPickle as pickle

from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


def fetchGenomePositions(genomeSequence): #, trainIndicatorSet, testIndicatorSet):
    '''Aim of function is to return the effective indicator sets for train/test, ie
    the subsets of the indicator sets consisting in the valid positions of the genomeSequence
    ie the ones holding A,C,G,T but not N (or any other letter).
    Returns these two subsets, one for train and one for test and a third array (boolean) containing the set 
    of valid positions (indicator array).'''
    
    L = len(genomeSequence)
#    #Check that genome seq and the indicator sets have the same size:
#    if L != trainIndicatorSet.shape[0] or L != testIndicatorSet.shape[0]:
#        
#        print "No-go: length of genome seq does not match one of the indicator sets!"
#        return
#        
#    trainIndicatorSetEffective = trainIndicatorSet.copy()       
#    testIndicatorSetEffective = testIndicatorSet.copy() 
        
    qualifiedPositions = np.ones(shape = L, dtype = 'int8')    
    for i in range(L):
        
        if genomeSequence[i] not in ('A', 'T', 'C', 'G'): #UPPER??
            
            qualifiedPositions[i] = 0
#            trainIndicatorSetEffective[i] = 0
#            testIndicatorSetEffective[i] = 0
            
#    return trainIndicatorSetEffective, testIndicatorSetEffective, qualifiedPositions
    return qualifiedPositions
        

#def getSamplingStats(rootOutput, lastRepeatNr, epochs, batchSize, nrSteps, qualifiedPositions, windowList, trainIndicatorSet, testIndicatorSet ):
def getSamplingStats(rootOutput, lastRepeatNr, epochs, batchSize, nrSteps, nrTestSamples, qualifiedPositions, windowList, testTrainRatio = 0.8): #, trainIndicatorSet, testIndicatorSet ):
    '''For each window length in the input list compute average and std dev of the number of sampled
    positions in all adjacent windows of that length (that can fit in the valid genome
    sequence). Returns the corresponding list of avg/std devs. Plots ...
    
    Obs: the use of the indicator sets have been out-commmented in the code; it is really not needed, since we only 
    need to care about whether the complete sampling over the genome sequence is uniform. If these two sets are not 
    sampled uniformly over the genome sequence, that non-uniformity may show up here. And should it be canceled out 
    it would be just fine. So all there is to check re the indicator sets is that they have the desired relative size 
    (and that could if wanted be seen right off their file sizes, as they are saved as sparse arrays).'''

    resultsDict = {}
    resultsDict['Train'] = {}
    resultsDict['Test'] = {}
    
##    #load the train/test indicator sets used in the sampling test:
##    loadFile = rootOutput + 'testOfSampling_LSTM_trainSetIndicator.p'
##    trainIndicatorSet = pickle.load(open( loadFile, "rb"))
#    lTrainInd = len(trainIndicatorSet) 
#    
##    loadFile = rootOutput + 'testOfSampling_LSTM_testSetIndicator.p'
##    testIndicatorSet = pickle.load(open( loadFile, "rb"))
#    lTestInd = len(testIndicatorSet)
#    
#    testTrainRatio = float(lTestInd)/lTrainInd
#    print "testTrainRatio in sampling: ", testTrainRatio

    repeatNrList = np.arange(0, lastRepeatNr + 1, step = 10)
    samplingArrayTrain = 0
    samplingArrayTest  = 0
    for repeatNr in repeatNrList[1:]:
        
        #load the sampling sets train/test for the repeats since last repeat-dumped-at:
        loadFile = rootOutput + 'testOfSampling_LSTM_samplingCountsTotal_train_repeatNr' + str(repeatNr) + '.p'
        trainSetTheseRepeats = pickle.load(open( loadFile, "rb"))

        loadFile = rootOutput + 'testOfSampling_LSTM_samplingCountsTotal_test_repeatNr' + str(repeatNr) + '.p'
        testSetTheseRepeats = pickle.load(open( loadFile, "rb"))

        #accumulate the samplings:
        samplingArrayTrain += trainSetTheseRepeats
        samplingArrayTest += testSetTheseRepeats

#        #First check that only the positions indicated for train and test are sampled (for these two, resp.):
#        intersectTrain = np.intersect1d(samplingArrayTrain, trainIndicatorSet)
#        intersectTest = np.intersect1d(samplingArrayTest, testIndicatorSet)
#        if len(intersectTrain) != len(samplingArrayTrain)  and len(intersectTest) != len(samplingArrayTest):
#            
#            print "Sampling only done within indicator sets."
#        
#        else:
#            
#            print "Warning: Sampling NOT only done within indicator sets!"
#            raw_input("Do you want to continue?")
#        
        
        #Next: To compute the stats we need to remove from each of the sampling sets: 
        #1) the non-qualified positions and 
        #2) the complement of the indicator set. 
        #It is ok to do this, since we have checked that no sampling has been done outside
        #the indicators/qualified:
        qual = np.where(qualifiedPositions > 0)[0]
#        print qual[:10]
#        trainInd = np.where(trainIndicatorSet > 0)
#        testInd = np.where(testIndicatorSet > 0)
        
#        qualAndTrainInd = np.intersect1d(qual, trainInd)
#        qualAndTestInd = np.intersect1d(qual, testInd)
#         
#        qualAndInd = [qualAndTrainInd, qualAndTestInd]

#        samplingArrays = [samplingArrayTrain.take(qualTrain), samplingArrayTest.take(qualTest)]
        samplingArrays = [samplingArrayTrain.toarray(), samplingArrayTest.toarray()]
#        print samplingArrayTrain.shape
#        print samplingArrayTest.shape
#        print samplingArrays[0].take(qual[:100])
#        print samplingArrayTrain.toarray().take(qual[:10])
    #    indictorArrays = [trainIndicatorSet, testIndicatorSet]
    
        sizeOfSamplingTrain = repeatNr*epochs*nrSteps*batchSize
        sizeOfSamplingTest = repeatNr*nrTestSamples
        sizeOfSampling = [sizeOfSamplingTrain, sizeOfSamplingTest]
        
        for i in range(2):
            
            samplingArray = samplingArrays[i][0]
    #        indictorArray = indictorArrays[i]
            
#            L = samplingArray.shape[0] + .0
#            L = len(qualAndInd[i])
            L = len(qual)
            
            for j in range(len(windowList)):
                        
                lWindow = windowList[j]
                if i == 0 and not(resultsDict['Train'].has_key(lWindow)):
                    resultsDict['Train'][lWindow] = []
                elif i == 1 and not(resultsDict['Test'].has_key(lWindow)):
                    resultsDict['Test'][lWindow] = []
                
                
                nrWins = int(np.floor(L/lWindow))
                print "nrWins: ", nrWins
                aim = float(sizeOfSampling[i])/nrWins #ideal nr of samples per window when uniformly sampled
                
                winSums = np.zeros(shape = nrWins)
                for k in range(nrWins):
#                    
##                tag nrWins lange vinduer af qualified; sml med qualTrain/qual
#                    winQualAndInd = qual[k*lWindow:(k+1)*lWindow] #set of lWindow indices in both qualified and train/test sets
#                    
                    winSums[k] = np.sum(samplingArray.take(qual[k*lWindow:(k+1)*lWindow]))
                
#                print qual[1*lWindow:(1+1)*lWindow]
#                winSums = np.sum([samplingArray.take(qualAndInd[i][k*lWindow:(k+1)*lWindow]) for k in range(nrWins)])
#                winSums = np.sum(np.asarray([samplingArray.take(qual[k*lWindow:(k+1)*lWindow]) for k in range(nrWins)]), axis = 1)
#                print winSums.shape
                
                avg = np.mean(winSums)
                std = np.std(winSums)
                
                if i == 0:
                    print "Train, repeat, window: ", repeatNr, lWindow
                    resultsDict['Train'][lWindow].append([repeatNr, avg, std, aim])
                elif i == 1:
                    print "Test, repeat, window: ", repeatNr, lWindow
                    resultsDict['Test'][lWindow].append([repeatNr, avg, std, aim])
                print "avg, std: ", avg, std
                
        
    #plot:
    colors = cm.get_cmap('Set3')
    for typeKey in resultsDict:
        
        plt.figure()
        cnt = 0
        for lWindow in resultsDict[typeKey]:
            
            vals = np.asarray(resultsDict[typeKey][lWindow]).T
            repeatNrs = vals[0]
            avgs = vals[1]
            stds = vals[2]
            aims = vals[3]
            
            plt.errorbar(repeatNrs, avgs, yerr = 10*stds, color = colors(cnt+2), linestyle = '', marker= 'o', label = 'Sampled, window: ' + str(lWindow))
            plt.plot(repeatNrs, aims, color = colors(cnt+2), linestyle = '--', label = 'Ideal, window: ' + str(lWindow))
            plt.legend()
            
            cnt +=1 
            
        plt.savefig(rootOutput + 'SamplingTest_' + typeKey + '.pdf')
        plt.close()
        
        
    return resultsDict
                
                
                
                
                

    
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:35:43 2019

@author: Christian Grønbæk
"""

'''
Contents:
-- likelihood ratio (LR) tests of NN-models vs frq-models. LR tests of frq-models against each other: see frqModels module.
-- the LR test done here follow Vuong's paper on Cox's tests; the type of test for non-nested models is applied.
    

Usage:
    
####################################################

Import module:

####################################################

import dnaNet_stats as stats


####################################################

Input data:

####################################################

# Human genome 

#rootGenome = r"/Users/newUser/Documents/clouds/Sync/Bioinformatics/various_python/DNA_proj/data/human/"

#On binf servers:
#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr17.fa"
fileGenome = rootGenome +fileName
    
rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
fileName = r"hg38_chr22.txt"
fileGenome = rootGenome +fileName


#Read in data from genome and get it encoded:
exonicInfoBinaryFileName = ''
chromoNameBound = 1000
startAtPosition = 0
endAtPosition = 3e9 #some big number
outputEncoded_b = 1
outputEncodedOneHot_b = 1
outputEncodedInt_b = 0
outputAsDict_b = 0
outputGenomeString_b =0
randomChromo_b = 0
avoidChromo = []


encodedGenomeData =  stats.dataGen.encodeGenome(fileName = fileGenome, 
                       exonicInfoBinaryFileName = exonicInfoBinaryFileName,
                       chromoNameBound = chromoNameBound, 
                       startAtPosition = startAtPosition,
                       endAtPosition = endAtPosition,
                       outputEncoded_b = outputEncoded_b,
                       outputEncodedOneHot_b = outputEncodedOneHot_b,
                       outputEncodedInt_b = outputEncodedInt_b,
                       outputAsDict_b = outputAsDict_b,
                       outputGenomeString_b = outputGenomeString_b,
                       randomChromo_b = randomChromo_b, 
                       avoidChromo = avoidChromo)
        
               
#Then encodedGenomeData has the structure: genomeSeq, repeatInfoSeq, exonicInfoSeq, qualInfoSeq, sampledPositionsIndicatorSeq 

#Get a (sub) set of the encoded data on which to get the models predictions, and, finally, to compute their generalized-LR test figure:
genSamplesAtRandom_b = 0 #!!!
cutDownNrSamplesTo = 1e9
flankSize = 200
#use these default settings:
labelsCodetype = 0
outputEncodedType = 'int8'
convertToPict_b = 0
shuffle_b = 0
inner_b = 1
shuffleLength = 5
augmentWithRevComplementary_b = 0


outputSamples = stats.dataGen.getAllSamplesFromGenome(encodedGenomeData = encodedGenomeData, genSamplesAtRandom_b = genSamplesAtRandom_b, cutDownNrSamplesTo = cutDownNrSamplesTo, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = flankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)

#Then outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean

#Model files are placed in:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/"

####################################################

Now get the predictions:

####################################################


modelFile = "modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr230"
modelFileNameNN = rootOutput + modelFile

genomeIdName = 'hg38_chr22'
leftRight_b = 1
customFlankSize = flankSize
computePredAcc_b = 0

predArrayNN, qualArrayNN, avgPredArrayNN = stats.predictAcrossGenome(modelFileName = modelFileNameNN, genomeIdName = genomeIdName, computePredAcc_b = computePredAcc_b, outputFromGetAllSamplesFromGenome = outputSamples, customFlankSize = customFlankSize, leftRight_b = leftRight_b, windowLength = 10000, stepSize = 2500, batchSize = 500)

 
#To get the accuracy of the loaded model across a stretch of the genome (you need the predictions across a stretch):
windowLength = 20
stepSize = 10
Fourier_b = 0
avgPred, args = stats.computeAccuracyOnSamples(modelFileName = modelFileNameNN, genomeIdName = genomeIdName, labelArray = outputSamples[1], repeatInfoArray = outputSamples[2], predictionArray = predArrayNN, qualifiedArray = qualArrayNN, windowLength = windowLength, stepSize = stepSize, Fourier_b = Fourier_b)
  

#Compute the auto corr in the avgpredArray;
autoCorrNN = stats.autoCorrAvgPred(avgPredArray = avgPredArrayNN, qualArray = qualArrayNN, maxHorizon = 1000)

  

#Get prediction array for k-mer model (predicted distributions) at the same positions as obtained in the predictions of the NN model:
#(use outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean)
 
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/'    

k=5
fileFrq = "frqModel_chr22_k" + str(5) + ".txt"
fileNameFrq = rootFrq + fileFrq

resDictFrq = stats.frqM.readResults(fileNameFrq)
getAllPositions_b = 0
samplePositions = outputSamples[4] 
sampledPositionsBoolean = outputSamples[5]   
predArrayFrq, qualArrayFrq, samplePositions, sampledPositionsBooleanFrq  = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, samplePositions = samplePositions, startAtPosition=startAtPosition, endAtPosition=endAtPosition, samplePositionsIndicatorArray = sampledPositionsBoolean, getAllPositions_b = getAllPositions_b, resultsDictFrqModel=resDictFrq, k=k, chromoNameBound = chromoNameBound)    

#if the sample postions differ, there's something rotten; this sum should be zero:
np.sum(sampledPositionsBoolean- sampledPositionsBooleanFrq)

#Get the accuracy of the loaded k-mer model across a stretch of the genome (you need the predictions across a stretch):
genomeIdName = 'hg38_chr22'
modelFileName_kMer = rootFrq + 'k5_mer'
windowLength = 20
stepSize = 10
Fourier_b = 0
avgPred, args = stats.computeAccuracyOnSamples(modelFileName = modelFileName_kMer, genomeIdName = genomeIdName, labelArray = outputSamples[1], repeatInfoArray = outputSamples[2], predictionArray = predArrayFrq, qualifiedArray = qualArrayFrq, windowLength = windowLength, stepSize = stepSize, Fourier_b = Fourier_b)

  
    
####################################################

log-likelihood ratio tests:

####################################################    

    
#Likelihood ratio test of NN model vs frq model.    
#Get: encodedGenomeData and the pred-arrays as above (all based on the very same genomic seq of course). Then go:
pVal, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, predArrayFrqModel = predArrayFrq, qualifiedArrayFrqModel = qualArrayFrq, k = k, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize)
   
   
#Likelihood ratio test of NN model vs Markov model. For the latter we first read in the prob's from
#Yuhu's run of AK's model:
#Read in prob's from external:
fileName = r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'
positionArrayMarkov, predArrayMarkov = stats.readInProbsFromExternal(fileName)
#We have to "standardize" these so that the arrays match those from the other model's indexing (here a NN-model).
#For this we mus use the same settings for the genome data as used for the oter model:
samplePositions = outputSamples[4] 
getAllPositions_b = 0
predArrayMarkov, qualArrayMarkov, samplePositionsMarkov = stats.getPredArrayFromExternal(fileGenome =fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition, samplePositions = samplePositions, getAllPositions_b=getAllPositions_b , positionsExternal= positionArrayMarkov, probsExternal= predArrayMarkov, k= k)
#Then run the LR-test:
pVal, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, predArrayFrqModel = predArrayMarkov, qualifiedArrayMarkov = qualArrayMarkov, k = k, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize)

   

   

#To test the calc of the generalized LR test, we apply it to the fra model for k = 4 and k = 5; then check if the test size (LR) is the same as 
#that computed by the LR test in the frqModels module. To do that we however have to get the prediction of the models for the whole length of the
genome:


rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/'    

file4 = "frqModel_chr22_k4.txt"
fileName4 = rootFrq + file4
resDictFrq4 = stats.frqM.readResults(fileName4)

file5 = "frqModel_chr22_k5.txt"
fileName5 = rootFrq + file5
resDictFrq5 = stats.frqM.readResults(fileName5)

#Get LR acc to std LR for nested models:
dfDiff = 3*stats.np.power(4,10) - 3*stats.np.power(4,8)
p, testFig = stats.frqM.logLikelihoodRatioTest(resDictFrq4, resDictFrq5, dfDiff)


#Get the LR computed by the code here (below); first get pred-arrays for the two k-mer models:
exonicInfoBinaryFileName = ''
#chromoNameBound = 10
startAtPosition = 1e6
endAtPosition = 1e8 #some big number
#outputAsDict_b = 0
#outputGenomeString_b =0
#randomChromo_b = 0
#avoidChromo = []

#We consider all positions:s
getAllPositions_b = 1
samplePositions = 0 #just a placeholder
samplePositionsIndicatorArray = 0 #just a placeholder
 
predArrayFrq4, qualArrayFrq4, samplePositions4, samplePositionsIndicatorArray4 = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, startAtPosition=startAtPosition, endAtPosition=endAtPosition, getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionIndicatorArray, resultsDictFrqModel=resDictFrq4, k=4)    
predArrayFrq5, qualArrayFrq5, samplePositions5, samplePositionsIndicatorArray5 = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, startAtPosition=startAtPosition, endAtPosition=endAtPosition, getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionIndicatorArray, resultsDictFrqModel=resDictFrq5, k=5)    

#If the sample positions differ, there's something rotten; these sums should be zero:
np.sum(samplePositions5 - samplePositions4)
np.sum(samplePositionIndicatorArray4 - samplePositionIndicatorArray5)



#Encoded the data:
exonicInfoBinaryFileName = ''
encodedGenomeData =  stats.dataGen.encodeGenome(fileName = fileGenome, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputEncoded_b = 1,
               outputEncodedOneHot_b = 1,
               outputEncodedInt_b = 0,
               outputAsDict_b = outputAsDict_b,
               outputGenomeString_b = outputGenomeString_b,
               randomChromo_b = randomChromo_b, 
               avoidChromo = avoidChromo)



pVal, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions4, samplePositionsIndicatorArray = samplePositionIndicatorArray5,  predArrayFrqModel = predArrayFrq4, qualifiedArrayFrqModel = qualArrayFrq4, k = 4, predArrayNN = predArrayFrq5, qualifiedArrayNN = qualArrayFrq5, flankSize = 5)



########################################################################################################
# One-off's: to get genome seq for assembly hg38 split over chromo's; read in probs from external source
########################################################################################################

isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/GCF_000001405.38_GRCh38.p12_genomic_filter.fna
import dnaNet_dataGen as dataGen
chromoNameBound = 1000
stats.dataGen.splitGeneomeInChromosomes(root = '/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/', genomeFileName = 'GCF_000001405.38_GRCh38.p12_genomic_filter.fna', genomeShortName = 'hg38', rootOut = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/', chromoNameBound = chromoNameBound)


#Read in prob's from external:
fileName = r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'
positionArrayMarkov, predArrayMarkov = stats.readInProbsFromExternal(fileName)

########################################################################################################
# SNPs
########################################################################################################

#Assumes that we have obtained a predArray (from some model).

import snpAnalysis as snp

#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'
chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'

snpInfoArray = snp.readSNPdata(rootSnp + fileName)

#NN model:
rootOutNN = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/" 
modelFileName = modelFileNameNN
genomeIdName = genomeIdName
snpIdName = '1000Gs'
predictionArray = predArrayNN
qualArray = qualArrayNN
title = 'LSTM flanks200'
probsAtSnp = snp.fetchProbsAtSnps(snpInfoArray = snpInfoArray, chrNr = chrNr, predictionArray = predictionArray, qualArray = qualArray, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, rootOut = rootOutNN)


#k-mer model:
rootOutFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/chr22/' 
modelFileName = modelFileName_kMer 
genomeIdName = genomeIdName
snpIdName = '1000Gs'
predictionArray = predArrayFrq
qualArray = qualArrayFrq
title = 'k5 mer'
probsAtSnp = snp.fetchProbsAtSnps(snpInfoArray = snpInfoArray, chrNr = chrNr, predictionArray = predictionArray, qualArray = qualArray, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, rootOut = rootOutFrq)


'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import frqModels as frqM

import dnaNet_dataGen as dataGen

from keras.models import model_from_json

import scipy.stats as stats

import cPickle as pickle

from scipy.fftpack import fft, ifft, dst, rfft

import csv




def predictAcrossGenome(modelFileName, 
                        outputFromGetAllSamplesFromGenome,
                        genomeIdName, #for saving the prediction array; could eg be a chromo name
                        customFlankSize = 50,
                        computePredAcc_b = 0, 
                        overlap = 0,
                        leftRight_b = 0,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence or for a set of sample positions (endcoded) by 
    using the input estimated/trained model.
    
    Input:
        
        modelFileName: name of the model file for the trained model (must have a .h5 companion)
        outputFromGetAllSamplesFromGenome: tuple as returned by dataGen.getAllSamplesFromGenome

        
        computePredAcc_b: boolean; whether or not (0) to compute the accuracy on predicting the base at each position provided. If using 
                          sampled positions not covering a contiguous part of the genome, the windowing does not really makes sense (so set 
                          windowLength = 1 and stpesize = 1)
        
        
    Returns: predArray, Q, S, avgPred where
    
    predArray: the predicted distribution of the four letters at each position in the input genomic data (encodedGenomeData)
    Q: array of booleans indicating for each position whether the sample is qualified or not (letter ACGT or not)
    S: indicator array of the actual genomic positions(1 means that the position was included; this refers to positions in the enocodedGenomeData) 
    avgPred: array of average accuracy obtained by averaging over windows (one average per step)
        
    '''
    
    net = model_from_json(open(modelFileName).read())
    net.load_weights(modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
    sizeOutput = net.output_shape[1]

    #Decipher the test data:
    #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
    Xt, Yt, Rt, Qt, St, St_boolean = outputFromGetAllSamplesFromGenome
    
    print("Length of genome seq read in: %d" % len(St))

    lSamples_X, doubleFlankSize, letterShape = Xt.shape 

    #for some models we have split the left and right flanks in the input the model:
    if leftRight_b == 1:
        
        Xt_lr = np.zeros(shape = (lSamples_X, 2, customFlankSize, letterShape)) 
        Xt_l = np.zeros(shape = (lSamples_X, customFlankSize, letterShape)) 
        Xt_r = np.zeros(shape = (lSamples_X, customFlankSize, letterShape))                
        
        for i in  range(lSamples_X): 
            Xt_left = Xt[i, :(customFlankSize + overlap) , :].copy()
            Xt_right = Xt[i, (customFlankSize - overlap):, :].copy()
            #and reverse it:
            Xt_right = np.flip(Xt_right, axis = 0)
            if i == 0:
                print Xt[i, (customFlankSize - overlap):, :][:5]
                print Xt_right[-5:]
            #Then
            Xt_lr[i][0] = Xt_left
            Xt_lr[i][1] = Xt_right
            
            Xt_l[i]= Xt_left
            Xt_r[i] = Xt_right
            
        print("Xt_lr shape ", Xt_lr.shape)



    lSamples = len(Yt)
    
    if lSamples != lSamples_X:
        print("The number of sequences does not match the number of labels! --- this is a no-go!!!")
    if len(Qt) != lSamples:
        print("The number of positions with a qualified-not qualified annotation does not match the number of labels! --- this is a no-go!!!")    
        
    
    #Call the prediction           
    if leftRight_b == 0:
        
        predArray = net.predict(Xt, batch_size = batchSize)
    
    elif leftRight_b == 1:
        
        predArray = net.predict([Xt_l, Xt_r], batch_size = batchSize)
    
    print("Nr of samples: %d ; of which are predicted: %d" % (lSamples, predArray.shape[0]))
    
    if lSamples != predArray.shape[0]:
        print("Nr predictions does not match the number of samples -- this is a no-go!! (implies that positions in prediction array do not corr to genome positions)")
    
        
    #Keep a copy of the results:
    dumpFile = modelFileName + '_' + 'predArray' + '_' + genomeIdName
    pickle.dump(predArray, open(dumpFile, "wb") )
    dumpFile = modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdName
    pickle.dump(Qt, open(dumpFile, "wb") )
            

    if computePredAcc_b == 1:
        
        avgPred = computeAccuracyOnSamples(modelFileName = modelFileName, 
                             labelArray = Yt,
                             repeatInfoArray = Rt,
                                       predictionArray = predArray,
                            qualifiedArray = Qt,
                            windowLength = windowLength,
                        stepSize = stepSize)
    
        return predArray, Qt, avgPred

    else:
        
        return predArray, Qt, [] #empty list just a placeholder
    
    
def computeAccuracyOnSamples(modelFileName, 
                             genomeIdName,
                             labelArray,
                             repeatInfoArray,
                                       predictionArray,
                            qualifiedArray,
                            windowLength = 100,
                        stepSize = 100,
                        Fourier_b = 0):
    
    lSamples = predictionArray.shape[0]
    
    nrSteps = int((lSamples - windowLength)/stepSize)
    
    print("lSamples ", lSamples,", nrSteps ", nrSteps )
    
    cntTot = 0
    cntCorr = 0.0
    cntCorrRep = 0.0
    cntTotRep  = 0.0
            
    windowList = []
    windowArray = np.zeros(shape = windowLength, dtype = 'float32')
    avgPred = np.zeros(shape = nrSteps, dtype = 'float32')
    
    Y = labelArray
    R = repeatInfoArray
    
    cntErr = 0
    qualified_b = 1
    for j in range(nrSteps):
        
        #first window: read in the following windoLength worth of sites:
        if j == 0:
            
            qualified_b = 1 #to record whether the window is qualified or not (ie contains a non-ACGt letter)
            for i in range(windowLength): #range(predArray.shape[0]):
                
#                print " ".join(map(str,Yt[i])), " ".join(map(str,predArray[i]))

                #Check if position is qualified; if not skip the sample:
                if qualifiedArray[i] == 0:
                    qualified_b = 0
                    continue
                
                predIdx = np.argmax(predictionArray[i])
                
#                print predIdx
                
                if Y[i][predIdx] > 0.5:
                    cntCorr += 1.0
                    if R[i] > 0.5:
                        cntCorrRep += 1.0
                        cntTotRep  += 1.0
                    windowList.append(1.0)
                    
                else:
                    windowList.append(0.0)
                    if R[i] > 0.5:
                        cntTotRep  += 1.0
                    
                cntTot += 1
            
            if qualified_b == 1:
                windowArray = np.asarray(windowList)
                avgPred[j] = np.mean(windowArray) #and j = 0
            else:
                avgPred[j] = 0.5
            
        else:
            
            qualified_b = 1 #to record whether the window is qualified or not (ie contains a non-ACGt letter)
            
            #remove first stepSize elt's from list
            for k in range(stepSize):
                try:
                    windowList.pop(0)
                except IndexError:
                    print j
                    cntErr += 1
                    
            #Append the stepSize next elts:
            for l in range(stepSize): 
                i = windowLength + (j-1)*stepSize + l
                
                #Check if position is qualified; if not skip the sample:
                if qualifiedArray[i] == 0:
                    qualified_b = 0
                    continue
                
                predIdx = np.argmax(predictionArray[i])
                if Y[i][predIdx] > 0.5:
                    cntCorr += 1
                    if R[i] > 0.5:
                        cntCorrRep += 1.0
                        cntTotRep  += 1.0
                    windowList.append(1.0)
                        
                else:
                    windowList.append(0.0)
                    if R[i] > 0.5:
                        cntTotRep  += 1.0
                
                cntTot += 1


            if qualified_b  == 1:
                windowArray = np.asarray(windowList)    
                avgPred[j] = np.mean(windowArray)
            else:
                avgPred[j] = 0.5
            
            
    plt.figure()       
    plt.plot(avgPred) 
    plt.savefig(modelFileName + '_' + genomeIdName + '_predPlot.pdf' )    
    
    
    #forcing the avgPred to be periodic (enabling the Fourier transform):
    avgPred[nrSteps-1] = avgPred[0]
    print("Avg pred at 0: %f  and at nrSteps: %f" %(avgPred[0], avgPred[nrSteps-1]) )
    
    if Fourier_b == 1:
        
        #Fourier transform it:
        fftAvgPred = rfft(avgPred) #scipy fast Fourier transform
        print fftAvgPred.shape[0]
        plt.figure()
        plt.title('fft avg prediction')  
        start = 0 #int(nrSteps/34)
        end = nrSteps - 1 #fftAvgPred.shape[0] -1 #int(nrSteps/33)
#        plt.bar(range(start,end),fftAvgPred[start:end]) 
        plt.bar(range(start,end), fftAvgPred[range(start,end)]) 
        #plt.plot(fftAvgPred)
        plt.savefig(modelFileName + '_FourierTransformPredPlot' + '_' + genomeIdName + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )

        args = np.argsort(fftAvgPred)
        print("Bottom 40 frq's (neg coeffs probably) ", args[:40])
        print(".. and their coeff's", fftAvgPred[args][:40])
        plt.figure()
        plt.title('fft, frqs vs coeffs, lowest 1000 coeffs')  
        plt.scatter(args[:1000], fftAvgPred[args][:1000])
        plt.savefig(modelFileName + '_FourierTransformPredPlot_lowestCoeffs' + '_' + genomeIdName + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
#        print "frqs, bottom ", args[:500]
        
        print("Top 40 frq's (pos coeffs probably) ", args[::-1][:40])
        print(".. and their coeff's", fftAvgPred[args[::-1]][:40])
        plt.figure()
        plt.title('fft, frqs vs coeffs, highest 1000 coeffs')  
        plt.scatter(args[::-1][1:1000], fftAvgPred[args[::-1]][1:1000])
        plt.savefig(modelFileName + '_FourierTransformPredPlot_highestCoeffs' + '_' + genomeIdName + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
#        print "frqs, top ", args[::-1][1:500]
        
        
    avgCorr = cntCorr/cntTot
    print("Average prediction acc : %f" % (avgCorr))

    nrReps = np.sum(R)
    if nrReps > 0.5: #ie if there are any repeats recorded
        avgCorrRep = cntCorrRep/cntTotRep
        avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
        print("Average prediction acc at repeats: %f and elsewhere: %f" % (avgCorrRep, avgCorrNonRep))
    else:
        print("No repeat sections were recorded in the genome data.")
    
    if Fourier_b == 1:
        return avgPred, args
    else:
        return avgPred,  []
    

def getPredArrayFrqModel(fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, samplePositions, samplePositionsIndicatorArray, getAllPositions_b , resultsDictFrqModel, k):

    '''
    
    Input:
        
    samplePositions: array of the positions of the samples; the samples must be generated witht the same data (fileGenome) etc --- same startAtPosition
    and endAtPosition. The sample positions will (then) be relative to the startAtPosition.       
    samplePositionsIndicatorArray: array of booleans (0/1) indicating whether the prediction should be called for a position (1) or not (0)     
    getAllPositions_b: if 1 the samplePositionIndicatorArray will consist of 1's for the length of the read-in genomic data (which is done by fastReadGenome)
    
    
    
    '''
    
    #Read the genome seq:
    Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileName = fileGenome, 
           exonicInfoBinaryFileName = exonicInfoBinaryFileName,
           chromoNameBound = chromoNameBound, 
           startAtPosition = startAtPosition,
           endAtPosition = endAtPosition,
           outputAsDict_b = 0,
           outputGenomeString_b = 0,
           randomChromo_b = 0, 
           avoidChromo = [])

    lenX = len(X)
    
    if getAllPositions_b == 1:
        
#        lenPred = lenX
#        nrSamples =  lenX - 2*k - 1
        samplePositions = np.arange(k, lenX - k - 1) #, dtype='int64')
#        for i in range(lenX):
#            samplePositions[i] = i
        
        samplePositionsIndicatorArray = np.ones(shape=lenX)
        #only the first and last part both of length flankSize are not "sampled":
        for i in range(k):
            samplePositionsIndicatorArray[i] = 0
            samplePositionsIndicatorArray[lenX - 1 - i] = 0
            
        lenPred = samplePositions.shape[0]
        
    else:
        print("getAllPositions_b is set to 0, so a samplePositionIndicatorArray must be provided")
    
        lenPred = samplePositions.shape[0]  #= number of samples (some may be repeated)
    
    predArray = np.zeros(shape=(lenPred, 4)) #data type is float64 by default
    Q = np.ones(shape = lenPred, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    
    #Loop through the genome seq, get the context at each position and look up the 
    #the predicted distr in the results of the frq model:
    cntMissingKeys = 0
    i = 0 #index in pred/Q array
    for idx in samplePositions:
        
#        if samplePositionsIndicatorArray[idx] == 0:
#            continue
        
        if idx < k:
            Q[i] = 0
            
        #Determine whether the position is qualified or not (: all ACGT letters in context and at the position or not), and record it in the assigned array:
        if X[idx-k:idx+k+1].count('W') > 0:
            Q[i] = 0
        
        context = X[idx-k:idx] + '*' + X[idx+1:idx+k+1]
    
        try:
            predArray[i] = resultsDictFrqModel[context][3:7]
        except KeyError:
            cntMissingKeys += 1
            #put in a random bid:
            randomIdx = np.random.randint(0,4) 
            predArray[i][randomIdx] = 1.0

        if i < 10:
            print idx
            print context
            print X[idx-k:idx+k+1]
            print predArray[i]
            
        i += 1
            
    print("cntMissingKeys: ", cntMissingKeys)
            
        
    return predArray, Q, samplePositions, samplePositionsIndicatorArray
    
    


def loglikelihoodRatioTestNonNestedModels(encodedGenomeData, samplePositions, predArrayFrqModel, qualifiedArrayFrqModel, k, predArrayNN, qualifiedArrayNN, flankSize):
    
    '''
    Computes the generalized LR test for non-nested models acc to Vuong/White/Cox test.
    
    Input:
        encodedGenomeData: one hot encoded genomic sequence.
#        samplePositionIndicatorArray: array of booleans (0/1) indicating whether a position in the encodedGenomeData is to be considered or not (ie whether it was sampled or not for the predictions)
        
        predArrayFrqModel: array of predicted distributions (for the occ of the four letters, ACGT) for the same genomic sequence for some model making predictions
        on contexts parameterized by a single parameter, k (eg a k-mer model)
        qualifiedArrayFrqModel: array of booleans (0/1) obtained together with the predArrayFrqModel, indicating whether the position is dis-qualified (0) or not (1), ie if the position contains a not-ACGT or an ACGT.
        predArrayNN: array of predicted distributions (for the occ of the four letters, ACGT) for the same genomic sequence for some model making predictions
        on contexts parameterized by a single parameter, flankSize (eg a NN convolution model taking contexts of flank size flankSize) 
        qualifiedArrayNN: as qualifiedArrayFrqModel, but for the N  model
        
    Output: p-vales, number of samples/contexts (genomic positions), log-likelihood ratio, corresponding variance 
        
    '''

    
    #We must compute:
    # the log-likelihood ratio (LR) of the two models (NN over frq model)
    # the variance corr to the LR (as in Vuong's paper)
    #Then 1/sqrt(n) * LR/variance is std normally distr and a p-value for the test-stat can easily be had 
    
    #Steps:
    #1. initalize LR = 0, varLR = 0
    #2. loop over all positions in genome
    #for each postion: 
    #compute l_NN = log(p_NN(base_i)) (if p_NN(base_i) == 0 set log(...) = 0; reason xlogx -> 0 when x -> 0)
    #compute l_frq = log(p_frq(base_i)) (if p_frq(base_i) == 0 set log(...) = 0; reason xlogx -> 0 when x -> 0)
    #add LR_i = l_NN - l_LR to LR and LR_i^2 to varLR
    #when loop complete: return LL, varLR
    
    #Obs: 
    #predArrayFrqModel starts at genome position k (the first left-hand flank is at 0,1 ..., k-1)
    #predArrayNN start at genome postion flankSize (the first left-hand flank is at 0,1 ..., flankSize-1)
    
    
    genomeSeq, repeatInfoSeq, exonicInfoSeq =  encodedGenomeData
#    
#    lenGenome = genomeSeq.shape[0]
    
    

    startAt = max(flankSize, k)
    
    print("First position to be considered: ", startAt)

    n = 0
    LR = 0.0
    v = 0.0
#    for i in range(startAt, lenGenome - startAt - 1):
    i = 0 #index in pred/Q arrays
    for idx in samplePositions:
        
#        if samplePositionsIndicatorArray[idx] == 0:
#            continue
        
        #Determine whether the position is qualified or not; if not skip the position:
        if qualifiedArrayFrqModel[i] == 0 or qualifiedArrayNN[i] == 0: #np.max(genomeSeq[i]) > 2 or 
            i += 1
            continue
        
        predFrq = predArrayFrqModel[i]
        predNN = predArrayNN[i]
        
        #To get the predicted probability of the base at position i:
        #-- the encoding has a 1 at the index of the base, and zeros elsewhere
        #-- so the dot product of the predicted distr with the encoding gives the probability acc to the model of the actual base
        dot_frq = np.dot(predFrq,genomeSeq[idx]) 
        if dot_frq > 0: 
            l_frq = np.log(dot_frq) 
        else:
            l_frq = 0
        dot_NN = np.dot(predNN,genomeSeq[idx])
        if dot_NN > 0:
            l_NN = np.log(dot_NN)
        else:
            l_NN = 0
        
        if i < 10: 
            print("base:",genomeSeq[idx])
            print("predFrq ", predFrq)
            print("at %d dot frq: %lf", idx, dot_frq)
            print("predNN ", predNN)
            print("at %d dot nn: %lf", idx, dot_NN)
            
        LR += l_NN - l_frq
        
        v += np.power(l_NN - l_frq, 2)
        
        i += 1
        n += 1
        
        
    print("The LR is found to be: ", LR)
    var = v/n - np.power(LR/n,2)
    print("The variance is found to be: ", var)
        
    testFig = LR/np.sqrt(n*var)
    
    #look-up the percentile:
    pVal = 1 - stats.norm.cdf(testFig, loc=0, scale=1)
    
    print("p-value of generalized log-likelihood test: ", pVal)
    
    return pVal, n, LR, var
        
        
        

def allInOneOnChromosomeList(genomeName, listOfChromosomeNames, flankSizeNN, flankSizeFrq, modelFile = "modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr230"):
    
    #Model files are placed in:
    rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/human/inclRepeats/"
    
    #Frq-model results are here:
    rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/firstRun/'    
        
    
    for chromoName in listOfChromosomeNames:

        rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/" + genomeName + "/"
        fileGenome = rootGenome + chromoName + ".fa"
                        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 10
        startAtPosition = 0
        endAtPosition = 3e9 #some big number
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b =0
        randomChromo_b = 0
        avoidChromo = []
        
        
        encodedGenomeData =  dataGen.encodeGenome(fileName = fileGenome, 
                       exonicInfoBinaryFileName = exonicInfoBinaryFileName,
                       chromoNameBound = chromoNameBound, 
                       startAtPosition = startAtPosition,
                       endAtPosition = endAtPosition,
                       outputEncoded_b = outputEncoded_b,
                       outputEncodedOneHot_b = outputEncodedOneHot_b,
                       outputEncodedInt_b = outputEncodedInt_b,
                       outputAsDict_b = outputAsDict_b,
                       outputGenomeString_b = outputGenomeString_b,
                       randomChromo_b = randomChromo_b, 
                       avoidChromo = avoidChromo)
        
        #Get a (sub) set of the encoded data on which to get the models predictions, and, finally, to compute their generalized-LR test figure:
        genSamplesAtRandom_b = 0 #!!!
        cutDownNrSamplesTo = 4e9
        #use these default settings:
        labelsCodetype = 0
        outputEncodedType = 'int8'
        convertToPict_b = 0
        shuffle_b = 0
        inner_b = 1
        shuffleLength = 5
        augmentWithRevComplementary_b = 0

        
        flankSize = max(flankSizeNN, flankSizeFrq)
        outputSamples = dataGen.getAllSamplesFromGenome(encodedGenomeData = encodedGenomeData, genSamplesAtRandom_b = genSamplesAtRandom_b, cutDownNrSamplesTo = cutDownNrSamplesTo, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = flankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)

        #Then outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean
    
        ####################################################
        
        #Now get the predictions:
        
        ####################################################
        
    
        modelFileName = rootOutput + modelFile
        
        leftRight_b = 1
        customFlankSize = flankSizeNN
        computePredAcc_b = 0
        
        predArrayNN, qualArrayNN, avgPredArrayNN = predictAcrossGenome(modelFileName = modelFileName, computePredAcc_b = computePredAcc_b, outputFromGetAllSamplesFromGenome = outputSamples, customFlankSize = customFlankSize, leftRight_b = leftRight_b, windowLength = 10000, stepSize = 2500, batchSize = 500)
          
        
        #To get the accuracy across the chromo:
        windowLength = 20
        stepSize = 10
        Fourier_b = 1
        avgPred, args = computeAccuracyOnSamples(modelFileName = modelFileName, labelArray = outputSamples[1], repeatInfoArray = outputSamples[2], predictionArray = predArrayNN, qualifiedArray = qualArrayNN, windowLength = windowLength, stepSize = stepSize, Fourier_b = Fourier_b)
         
    
        k = flankSizeFrq
        fileFrq = chromoName + "_k" + str(k) + ".txt"
        fileNameFrq = rootFrq + fileFrq
        
        resDictFrq = stats.frqM.readResults(fileNameFrq)
        getAllPositions_b = 0
        samplePositions = outputSamples[4] 
        sampledPositionsBoolean = outputSamples[5]   
        predArrayFrq, qualArrayFrq, samplePositions, sampledPositionsBooleanFrq  = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, samplePositions = samplePositions, startAtPosition=startAtPosition, endAtPosition=endAtPosition, samplePositionsIndicatorArray = sampledPositionsBoolean, getAllPositions_b = getAllPositions_b, resultsDictFrqModel=resDictFrq, k=k)    
        
    
        ####################################################
        
        #log-likelihood ratio tests:
        
        ####################################################    
        
            
        #Get: encodedGenomeData and the pred-arrays as above (all based on the very same genomic seq of course). Then go:
        pVal, n, LR, var = loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, samplePositionsIndicatorArray = sampledPositionsBoolean , predArrayFrqModel = predArrayFrq, qualifiedArrayFrqModel = qualArrayFrq, k = k, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize)




def autoCorrAvgPred(avgPredArray, windowSize, stepSize, qualArray, maxHorizon = 1000):
    
    '''Computes the autocorrelation of the input prediction array (as output by predictAcrossGenome and getPredArrayFrq)
    up to a max horizon'''
    
    L = avgPredArray.shape[0]
    
    autoCorrArray = np.zeros(shape = maxHorizon)
    
    for k in range(maxHorizon):
        
        N_k = 0
        
        for i in range(L - k):
            
            pos = i*stepSize 
            
            if np.min(qualArray[pos:(pos+windowSize)]) == 0:
                continue
            
            if np.min(qualArray[(pos+k):(pos +k +windowSize)]) == 0:
                continue
            
            autoCorrArray[k] += avgPredArray[i]*avgPredArray[i+k]
            
            N_k += 1
            
        
        autoCorrArray[k] = autoCorrArray[k]/N_k
        
    return autoCorrArray



def readInProbsFromExternal(fileName):
    
    '''Read in prob's of ACGT (in that order) at every position in some genomic 
    sequence (generated b some model).
    
    The external file is to be tab-separated (.csv) with fields:
         
    #CHROM  POS     ID      REF     ALT     QUAL    FILTER INFO
    
    where a line looks like this:
        
    22      10510015        GAATTCTTGTGTTTATATAATAAGATGTC   A       T       0.294502        no      E=0.059693;P=0.294502,0.164261,0.198916,0.3423212    

    '''
    
    posList = []
    probsList = []
    
    with open(fileName, mode='r') as probsFile:
        
        probsReader = csv.reader(probsFile, delimiter='\t')
        
        lineCnt =  0
        for row in probsReader:
            
            if lineCnt < 2:
                lineCnt +=1
                continue
            
            E_P = row[7].split(';') #splits the info in a list with two entries: 'E= ..." and "P=..."            
            PstringRaw = E_P[1] 
            #fetch the part of the Pstring after "P=":
            Pstring = PstringRaw.partition('P=')[2]
            #split the Pstring
            PstringSplit = Pstring.split(',')
#            print PstringSplit
            Pvalues = map(float, PstringSplit)
            
            posList.append(int(row[1]))
            probsList.append(Pvalues)                        
            
            lineCnt +=1
        
        
    return np.asarray(posList), np.asarray(probsList)
    
    
    
def getPredArrayFromExternal(fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, samplePositions, getAllPositions_b , positionsExternal, probsExternal, k):

    '''
    Similar to getPredArrayFrqModel. Here though the prob's of the four base at every position are written
    in from an external source (by readInProbsFromExternal). And we fetch these prob's at a desired set of samplePositions.
    
    Input:
        
    samplePositions: set of positions at which we want to have the predictions (usually had from getting the predictions from an internal model, ie a NN model)
    coveredPositions: array of the positions at which the probabilities are had; these must be generated on the same data as referred to by fileGenome
    getAllPositions_b: if 1 the samplePositionIndicatorArray will consist of 1's for the length of the read-in genomic data (which is done by fastReadGenome)
    
    Returns: 
    
    predArray, qualArray, samplePositions

    
    
    '''
    
    if getAllPositions_b == 1:
        
        #Read the genome seq:
        Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileName = fileGenome, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = 0,
               outputGenomeString_b = 0,
               randomChromo_b = 0, 
               avoidChromo = [])
    
        lenX = len(X)

#        lenPred = lenX
#        nrSamples =  lenX - 2*k - 1
        samplePositions = np.arange(k, lenX - k - 1) #, dtype='int64')
#        for i in range(lenX):
#            samplePositions[i] = i
            
        lenPred = samplePositions.shape[0]
        
    else:
        print("getAllPositions_b is set to 0, so an array of samplePositions must be provided")
    
        lenPred = positionsExternal.shape[0]  #= number of samples (some may be repeated)
    
    predArray = np.zeros(shape=(lenPred, 4)) #data type is float64 by default
    Q = np.ones(shape = lenPred, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    
    #Loop through the genome seq, get the prob's at each position covered:
    i = 0 #index for pred/Q arrays
    for idx in samplePositions:
        
        if not(np.isin(idx, positionsExternal)):        

            Q[i] = 0
             #put in a random bid:
            randomIdx = np.random.randint(0,4) 
            predArray[i][randomIdx] = 1.0
            
        else:

            predArray[i] = probsExternal[i]
        
        i += 1
        
    return predArray, Q, samplePositions

############################################################################################################
##Pseudo code
############################################################################################################
    
    
def logLikelihoodNNmodel(NNmodelFile, genomeString):
    
    #load the NN model
    #initalize LL = 0
    #loop over all positions in genome
    #for each postion: 
    #get the prediction of the NN-model at the position
    #add log(p_model(base_i)) to LL
    #when loop complete: return LL

    pass

def logLikelihoodRatioNNmodelVSfrqModel(NNmodelFile, genomeString):
    
    #load the NN model
    #initalize LL = 0
    #loop over all positions in genome
    #for each postion: 
    #get the prediction of the NN-model at the position
    #add log(p_model(base_i)) to LL
    #when loop complete: return LL
    
    pass
    
    
def logLikelihoodRatioVariance(resultsDictFrqModel, k, NNmodelFile, flankSize, genomeString):
    
    #load the NN model
    #loop over all positions in genome
    #for each postion: 
    #1) fetch the context and look up the distr in the frqModel dict
    #2) get the prediction of the NN-model at the position
    #compute the variance term log(p_NN(b)/p_frq(b))^2 where b is the base at the position

    pass
    
def genLoglikelihoodRatioTest_pseudo(resultsDictFrqModel, k, NNmodelFile, flankSize, genomeString):
    
    #We must compute:
    # the log-likelihood ratio (LR) of the two models (NN over frq model)
    # the variance corr to the LR (as in Vuong's paper)
    #Then 1/sqrt(n) * LR/variance is std normally distr and a p-value for the test-stat can easily be had 
    
    #Steps:
    #1. load the NN model
    #2  initalize LR = 0, varLR = 0
    #3. loop over all positions in genome
    #for each postion: 
    #get the prediction of the NN-model at the position
    #compute l_NN = log(p_NN(base_i)) (if p_NN(base_i) == 0 set log(...) = 0; reason xlogx -> 0 when x -> 0)
    #get the prediction of the frq-model at the position: look-up the context in the dictionary (resultsDictFrqModel) 
    #compute l_frq = log(p_frq(base_i)) (if p_frq(base_i) == 0 set log(...) = 0; reason xlogx -> 0 when x -> 0)
    #add LR_i = l_NN - l_LR to LR and LR_i^2 to varLR
    #when loop complete: return LL, varLR

    
    net = model_from_json(open(modelFileName).read())
    net.load_weights(modelFileName +'.h5')
    

    genomeSeq, repeatInfoSeq, exonicInfoSeq =  genomeData

    lGenome = len(genomeSeq)
    
    print("Genome length: %d" % lGenome)


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:35:43 2019

@author: Christian Grønbæk
"""

'''
The code below -- all in a 'research state' -- was used for analysis downstream of the training of the neural networks 
as reported in the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 

The 'analysis downstream of the training ...' comprises:
* the 'prediction': applying the nets to the full genomes,
* calculating the performance (accuracy of the performance) and making all accompanying plots and tables 
associated etc
* making the likelihood-ratio (LR) tests of the neural networks vs the simpler benchmark models (frq-models; 
see frqModels module)
* all the Fourier transformation matter
* the SNP analyses
Plus much else.

Notes:

    -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
    -- the usage section below is split into eight parts; some of the calls/examples lie quite far in the past and 
        may not work as they are found here (this goes in particular for Part I and Part II). So ...
    -- OBS! Generally many of the example function calls in the 'Usage' below may be out-dated, so may take some 
        correction for it to run (functions  have changed during the development of it all), some may even be lost cases!
    -- What to do if that happens: Check the definition and maybe even the code of the function in the call; see if the
        argemnts to the function are all set and appropriately
    -- The prediction (predictOnChromosomes, predictAccrossGenomeInSegments and others) generally takes long time -- maybe confine it to a single chromosome for a start. Also, an LSTM on 
       flanks 200 take about four times as long to run (or more) than an other wise identical LSTM on flanks of size 50 ... 
    -- benchmark models: if you want to use one of the frq-models or the Markov model, please contact one of us (see frqModels module). This
        input is taken from toher code that is not included here.
    -- likelihood ratio (LR) tests of NN-models vs frq-models and Markov model. LR tests of frq-models against each other: 
        see frqModels module.
    -- the LR tests done here follow Vuong's paper on Cox's tests; the type of test for non-nested models is applied.
    
    
##################################################################################################
# Usage:
##################################################################################################


The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.

In the remaining part of this usage section, right below, you'll a find a lot of calls to some of the functions 
in this dataGen module. And many variations on setting the actual arguments' for some of the function calls (e.g
predictOnChromosomes, and you may first see a long list of various possible argument values, much of it for handling 
the genome files of various organisms or for several similar function calls on different models). Many of the main
functions of the other modules are called, eg from dnaNet_dataGen.

####################################################

# Import module:

####################################################

import dnaNet_stats as stats


####################################################

# Input data (just a list, mostly for convenience):

####################################################

# Human genome 

#rootGenome = r"/Users/newUser/Documents/clouds/Sync/Bioinformatics/various_python/DNA_proj/data/human/"

#On binf servers:
#single chromo, hg19
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr17.fa"
fileGenome = rootGenome +fileName
    
rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

#single chromo, hg18
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg18/"
fileName = r"hg18_chr12.fa"
fileGenome = rootGenome +fileName
 

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
fileName = r"hg38_chr22.txt"
fileGenome = rootGenome +fileName

#Yeast genome

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"
fileGenome = rootGenome +fileName

#Single chromo:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"R64_chr1.txt"
fileGenome = rootGenome +fileName

#Mouse:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
fileName =  r"Mus_musculus.GRCm38.dna_sm.primary_assembly.fa"
fileGenome = rootGenome +fileName

#Mouse, next-to-last ref assembly, mm9:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/"
fileName =  r"mm9.fa"
fileGenome = rootGenome +fileName






####################################################

#Part 1: 

#get predictions on a sample of a chromosome, for internal 
#(NN) or external model. Allows to do LR test on this sample.

####################################################


_________________________________________________________

Step1. Read in genome data and get a set of samples of it
_________________________________________________________

#Read in data from genome and get it encoded:
exonicInfoBinaryFileName = ''
chromoNameBound = 1000
startAtPosition = 0
endAtPosition = 3e9 #some big number
outputEncoded_b = 1
outputEncodedOneHot_b = 1
outputEncodedInt_b = 0
outputAsDict_b = 0
outputGenomeString_b = 1 #!!!
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
        
               
#Then encodedGenomeData has the structure: encoded genomeSeq , repeatInfoSeq, exonicInfoSeq, genomeSequence, effectiveStartAtPosition, effectiveEndAtPosition:
genomeSeq , repeatInfoSeq, exonicInfoSeq, genomeSequenceString, effectiveStartAtPosition, effectiveEndAtPosition = encodedGenomeData

#Get a (sub) set of the encoded data on which to get the models predictions, and, finally, to compute their generalized-LR test figure:
genSamplesAtRandom_b = 0 #!!!
startPosition = 10e6
endPosition = 11e6
cutDownNrSamplesTo = 1e26
flankSize = 200
#use these default settings:
labelsCodetype = 0
outputEncodedType = 'int8'
convertToPict_b = 0
shuffle_b = 0
inner_b = 1
shuffleLength = 5
augmentWithRevComplementary_b = 1 #!!!


outputSamples = stats.dataGen.getAllSamplesFromGenome(encodedGenomeData = encodedGenomeData, startPosition = startPosition, endPosition = endPosition, genSamplesAtRandom_b = genSamplesAtRandom_b, cutDownNrSamplesTo = cutDownNrSamplesTo, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = flankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)

#Then outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean, augWithRevCmpl_b

_________________________________________________________

Step 2. Get the predictions on the samples for some model, NN or external. Compute accuracy.
_________________________________________________________


#Model files is placed in:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/"

modelFile = "modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr230"
modelFileNameNN = rootOutput + modelFile

genomeIdName = 'hg38_chr22'
leftRight_b = 1
customFlankSize = flankSize
computePredAcc_b = 0

predArrayNN, qualArrayNN, avgPredArrayNN = stats.predictOnGenomeSamples(modelFileName = modelFileNameNN, genomeIdName = genomeIdName, computePredAcc_b = computePredAcc_b, outputFromGetAllSamplesFromGenome = outputSamples, customFlankSize = customFlankSize, leftRight_b = leftRight_b, windowLength = 10000, stepSize = 2500, batchSize = 500)

 
#To get the accuracy of the loaded model across a stretch of the genome (you need the predictions across a stretch):
windowLength = 20
stepSize = 10
Fourier_b = 0
avgPred, args = stats.computeAccuracyOnSamples(modelFileName = modelFileNameNN, genomeIdName = genomeIdName, labelArray = outputSamples[1], repeatInfoArray = outputSamples[2], predictionArray = predArrayNN, qualifiedArray = qualArrayNN, windowLength = windowLength, stepSize = stepSize, Fourier_b = Fourier_b)
  

#Compute the auto corr in the avgpredArray;
autoCorrNN = stats.autoCorrAvgPred(avgPredArray = avgPredArrayNN, qualArray = qualArrayNN, maxHorizon = 1000)

  

#Get prediction array for k-mer model (predicted distributions) at the same positions as obtained in the predictions of the NN model:
#(use outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean, augWithRevCmpl_b)
 
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/chr22/'    

k=5
fileFrq = "frqModel_chr22_k" + str(5) + ".txt"
fileNameFrq = rootFrq + fileFrq

resDictFrq = stats.frqM.readResults(fileNameFrq)
getAllPositions_b = 0
samplePositions = outputSamples[4] 
sampledPositionsBoolean = outputSamples[5]
sampleGenomeSequence =  encodedGenomeData[3]  
predArrayFrq, qualArrayFrq, samplePositions, sampledPositionsBooleanFrq  = stats.getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, sampleGenomeSequence = sampleGenomeSequence, fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, samplePositions = samplePositions, startAtPosition=startAtPosition, endAtPosition=endAtPosition,  resultsDictFrqModel=resDictFrq, k=k, chromoNameBound = chromoNameBound)    
                                                                                  

#if the sample positions differ, there's something rotten; this sum should be zero:
np.sum(sampledPositionsBoolean- sampledPositionsBooleanFrq)

#Get the accuracy of the loaded k-mer model across a stretch of the genome (you need the predictions across a stretch):
genomeIdName = 'hg38_chr22'
modelFileName_kMer = rootFrq + 'k5_mer'
windowLength = 20
stepSize = 10
Fourier_b = 0
avgPred, args = stats.computeAccuracyOnSamples(modelFileName = modelFileName_kMer, genomeIdName = genomeIdName, labelArray = outputSamples[1], repeatInfoArray = outputSamples[2], predictionArray = predArrayFrq, qualifiedArray = qualArrayFrq, windowLength = windowLength, stepSize = stepSize, Fourier_b = Fourier_b)

From here it is possible to run a LR-test of the NN-modle and the frq-model 
(for which the predictions are now had on the same sample of genomic positions). 
It is IMPORTANT to keep the genomic info fixed throughout, ie organism, chromosome, 
and startPosition/endPosition!

However, LR tests and more is done better via cutting up a longer genomic seq in
segments. Done below.

  
########################################################################
  
# Part 2. Prediction in segments. LR tests, Plotting   
  
#Obs: See Part 5 for 'big scale' LR tests/plotting
   
########################################################################
  
_________________________________________________________

Step1. Read in genome data 
_________________________________________________________
  
  
#Read in data from genome and get it encoded:
exonicInfoBinaryFileName = ''
chromoNameBound = 100
startAtPosition = 10500000 #15100000 
endAtPosition = 1e8 #some big number if not really specified
outputEncoded_b = 1
outputEncodedOneHot_b = 1
outputEncodedInt_b = 0
outputAsDict_b = 0
outputGenomeString_b = 1 #!!!
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
                       
genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString  =  encodedGenomeData 
      
IMPORTANT: keep the genomic info fixed throughout the steps below, ie organism, chromosome, 
and startPosition/endPosition! If you save the results, save this genmomic info too, so that
it's possible to redo!
     
_________________________________________________________

Step 2. Get the predictions on the loaded genomic sequence (or part of it)
        for some model, NN or external. Done by segmenting the genomic seq.
_________________________________________________________
     
  
#Model files is placed in:

#LSTM1:
rootOutput = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/'
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"

#LSTM4 on hg19
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/"
modelFileNameNN = 'modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150'


augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize = 200
computePredAcc_b = 1
Fourier_b = 0

#Get the predictions of the model for each segment of a set length:
segmentLength = 1000000
startAtSegment = 0

batchSize = 500
windowLength = 5
stepSize = 1
Fourier_b = 0
on_binf_b = 1

chromoName = 'hg38_chr22'
genomeIdName = chromoName + '_seg' + str(int(segmentLength))


# Now get the preditions in segments. Only run this if you haven't already!
modelFileName = modelFileNameNN4
stats.predictAccrossGenomeInSegments(rootOutput = rootOutput, rootModel = rootModel, modelFileName = modelFileName, genomeIdName = genomeIdName, encodedGenomeData = encodedGenomeData, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b,
                        startAtSegment = startAtSegment, 
                        overlap = 0,
                        leftRight_b = leftRight_b,
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = 1)  
                        
   
#To obtain a complete prediction array across all segments (that we have just produced -- so based on the same encodedData as when running the prediction across the genome (step1))
nrSegments = 100 #just higher than actual max segment nr
genomeIdName = chromoName  
rootOutput = rootOutput + r'/' + genomeIdName + r'/'                 
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileNameNN, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength)                     

_________________________________________________________

Step 3. Likelihood ratio tests:
_________________________________________________________

    
#Likelihood ratio test of NN model vs frq model.    
#Get: encodedGenomeData and the pred-arrays as above (all based on the very same genomic seq of course). 
predArrayNN, qualArrayNN, labelArrayNN, sampledPositionsNN, sampledPositionsBooleanNN = predArray, qualArray, labelArray, sampledPositions, sampledPositionsBoolean
#For the frq model use the sampledPositions from the NN model:
getAllPositions_b = 0
samplePositions = sampledPositionsNN
labelArray = labelArrayNN
samplePositionsIndicatorArray = sampledPositionsBooleanNN
sampleGenomeSequence = encodedGenomeData[3] # genomeSeqString is 3rd entry of encodedGenomeData
k = 4
file = "frqModel_k" + str(k) + ".txt"
rootFrq = '/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/'    
fileName = rootFrq + file
resDictFrq = stats.frqM.readResults(fileName)
predArrayFrq, qualArrayFrq, samplePositions, samplePositionsIndicatorArray, corrLabelArray = stats.getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence, labelArray = labelArray,  fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel=resDictFrq, k=k)    
#Then go:
subSampleFraction = 0.1
subSampleSize = 100000 #set to 0 if you want a subsample at the size of subSampleFraction of all positions to be used
pVal, testFig, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, predArrayFrqModel = predArrayFrq, qualifiedArrayFrqModel = qualArrayFrq, k = k, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize, subSampleFraction = subSampleFraction, subSampleSize = subSampleSize)

    
   
#Likelihood ratio test of NN model vs Markov model. For the latter we first read in the prob's from
#Yuhu's run of AK's model:
#Read in prob's from external: 
fileName = r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'
positionArrayMarkov, predArrayMarkov, refBaseListMarkov = stats.readInProbsFromExternal(fileName)
#We have to "standardize" these so that the arrays match those from the other model's indexing (here a NN-model).
#For this we must use the same settings for the genome data as used for the other model:
samplePositions = samplePositionsNN
labelArray = labelArrayNN
getAllPositions_b = 0
sampleGenomeSequence = encodedGenomeData[3] # genomeSeqString is 3rd entry of encodedGenomeData
k = -1 #placeholder
displacementOfExternal = 1 #indexing starts from 1 in results from Markov model
predArrayMarkov_shared, qualArrayMarkov_shared, samplePositionsMarkov_shared, sampleGenomeSequenceEncoded_shared = stats.getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, labelArray = labelArray, sampleGenomeSequence = sampleGenomeSequence, positionsExternal= positionArrayMarkov, probsExternal= predArrayMarkov, refBaseListExternal = refBaseListMarkov,  fileGenome =fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= k, displacementOfExternal = displacementOfExternal)
#Then run the LR-test (obs: samplePositionsMarkov_shared == samplePositionsNN by construction):
subSampleFraction = 0.1
subSampleSize = 100000 #set to 0 if you want a subsample at the size of subSampleFraction of all positions to be used
pVal, testFig, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, predArrayFrqModel = predArrayMarkov_shared, qualifiedArrayFrqModel = qualArrayMarkov_shared, k = k, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize, subSampleFraction = subSampleFraction, subSampleSize = subSampleSize)

        


#LR test of 1st vs 2nd NN:


#LSTM1 w/wo avg on rev compl:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/LSTM1/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/"
modelFileNameNN2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"

#LSTM2
#rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/LSTM2/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/"
#modelFileNameNN2 ="modelLSTM__2Conv2LayerLstm_flanks200_win3_filters64And256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr200"

#LSTM3
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/LSTM3/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/"
modelFileNameNN3 = 'modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr253'

#LSTM4: flanks 50, trained on hg19
rootOu tput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/LSTM4/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/"
modelFileNameNN4 = 'modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150'


genomeIdName = 'hg38_chr22_seg1e6'
augmentWithRevComplementary_b = 0
leftRight_b = 1
customFlankSize = 200
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1000000

#Only run this if you haven't already!
stats.predictAccrossGenomeInSegments(rootOutput = rootOutput, rootModel = rootModel, modelFileName = modelFileNameNN2, genomeIdName = genomeIdName, encodedGenomeData = encodedGenomeData, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b,
                        batchSize = 500,
                        windowLength = 5,
                        stepSize = 1,
                        Fourier_b = Fourier_b,
                        on_binf_b = 1)  

predArrayNN2, labelArrayNN2, qualArrayNN2, sampledPositionsNN2, sampledPositionsBooleanNN2 = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileNameNN2, genomeIdName = genomeIdName, encodedGenomeData = encodedGenomeData, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength)                     

#For the frq model use the sampledPositions from the NN model:
samplePositions = samplePositionsNN
samplePositionsIndicatorArray = sampledPositionsBooleanNN
#Then go:
pVal, testFig, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, predArrayFrqModel = predArrayNN2, qualifiedArrayFrqModel = qualArrayNN2, k = 0, predArrayNN = predArrayNN, qualifiedArrayNN = qualArrayNN, flankSize = customFlankSize)


************************************************************************************
* Digression: Test of LR method
************************************************************************************
#To test the calc of the generalized LR test: we apply it to the fra model for k = 4 and k = 5; then check if the test size (LR) is the same as 
#that computed by the LR test in the frqModels module. To do that we however have to get the prediction of the models for the whole length of the 
#genome (chromo):


rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/chr22/'    

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
chromoNameBound = 10
startAtPosition = 1e6
endAtPosition = 1e8 #some big number
#outputAsDict_b = 0
#outputGenomeString_b =0
#randomChromo_b = 0
#avoidChromo = []

#We consider all positions for the larger k (if we call that K we'll there have pred's for pos's K, K+1, ... len(genomeSeq) - K +1, while for the smaller k (k) we'll have
pred's for k, k+1, ... and we will only be able to compare at thier intersection)
getAllPositions_b = 1
samplePositions = 0 #just a placeholder
samplePositionsIndicatorArray = 0 #just a placeholder

predArrayFrq5, qualArrayFrq5, samplePositions5, samplePositionsIndicatorArray5 = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, resultsDictFrqModel=resDictFrq5, k=5)    
 
#Now get the pred's for the smaller k (4) on the positions covered for the larger k (5):
getAllPositions_b = 0
samplePositions = samplePositions5
samplePositionsIndicatorArray = samplePositionsIndicatorArray5
predArrayFrq4, qualArrayFrq4, samplePositions4, samplePositionsIndicatorArray4 = stats.getPredArrayFrqModel(fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, resultsDictFrqModel=resDictFrq4, k=4)    

#If the sample positions differ, there's something rotten -- by construction these sums should be zero:
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



pVal,  testFig, n, LR, var = stats.loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions4,  predArrayFrqModel = predArrayFrq4, qualifiedArrayFrqModel = qualArrayFrq4, k = 4, predArrayNN = predArrayFrq5, qualifiedArrayNN = qualArrayFrq5, flankSize = 5)

* test of LR-test done
************************************************************************************ 

_________________________________________________________

Step 4. Plots of the prediction arrays:
_________________________________________________________

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"


#Plots of models' probability of the reference (at each position)
#First we need the array of the reference on the very same positions as those predicted; the encoded genomic sequence is
#encodedGenomeData[0] where encodedGenomeData was generated by start/endPosition by dataGen.encodeGenome above; the
#corr genome string is encodedGenomeData[3]  
#the first position sampled is
fromPos = sampledPositionsNN[0]
#and the last (: the last position not included)
toPos = sampledPositionsNN[sampledPositionsNN.shape[0]-1] +1
#both relative to startPosition
#So:
labelArray =  encodedGenomeData[0][fromPos:toPos]
#Markov vs best NN
stats.plotRefPredVsRefPred(labelArray=labelArray,  rootOutput = rootOutput, predArray1 = predArrayNN , predArray2 = predArrayMarkov_shared, qualArrayShared = qualArrayMarkov_shared, modelName1 = 'LSTM1', modelName2 = 'Markov', bins = 50)

#Frq  vs best NN
qualArrayFrqNN_shared = qualArrayFrq*qualArrayNN[:qualArrayFrq.shape[0]]
stats.plotRefPredVsRefPred(labelArray = labelArray, rootOutput = rootOutput, predArray1 = predArrayNN , predArray2 = predArrayFrq, qualArrayShared = qualArrayFrqNN_shared, modelName1 = 'LSTM1', modelName2 = 'frq_k5', bins = 50)

#NN2 vs best NN
stats.plotRefPredVsRefPred(labelArray=labelArray,  rootOutput = rootOutput, predArray1 = predArrayNN , predArray2 = predArrayNN2, qualArrayShared = qualArrayNN2, modelName1 = 'LSTM1', modelName2 = 'LSTM1noAvg', bins = 50)


#Plots of model's "own confidence"
#Markov vs best NN
stats.plotMaxPredVsMaxPred(rootOutput = rootOutput, predArray1 = predArrayNN , predArray2 = predArrayMarkov_shared, qualArrayShared = qualArrayMarkov_shared, modelName1 = 'LSTM1', modelName2 = 'Markov',  bins = 50)

#Frq vs best NN
qualArrayFrqNN_shared = qualArrayFrq*qualArrayNN[:qualArrayFrq.shape[0]]
stats.plotMaxPredVsMaxPred(rootOutput = rootOutput, predArray1 = predArrayNN , predArray2 = predArrayFrq, qualArrayShared = qualArrayFrqNN_shared, modelName1 = 'LSTM1', modelName2 = 'frq_k4', bins = 50)



#######################################################################

# Part 3. Big runs of prediction in segments: across list of chromo's

#######################################################################

_________________________________________________________

Primary use: run on trained NN's
_________________________________________________________

#Human

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"


#LSTM1: w/wo avg on rev compl:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"

#LSTM11: as LSTM1 but at earlier training stage:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr15"



#LSTM2: as LSTM1 but not trained w aug rev compl:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM2/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM2/"
modelFileNameNN ="modelLSTM__2Conv2LayerLstm_flanks200_win3_filters64And256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr196"


#LSTM4: flanks 50, trained on hg19
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/"
modelFileNameNN = 'modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150'

#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'


#LSTM4S; trained on all odd numbered hg38-chromos; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr184'


#LSTM4S2; id to LSTM4S, trained on all odd numbered hg38-chromos, but stopped early; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S2/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S2/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr37'


#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

#LSTM4 on hg18 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'


#LSTM4P on part2 (test part) of the 80/20-split genome (GRCh38):
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/split/part2/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199'

#LSTM4 (!) on part2 (test part) of the 80/20-split genome (GRCh38):
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/split/part2/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl_part2/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr175'


#--------------


augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize =50
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1000000

batchSize = 528
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1

#start positions
#chr13: start 16000000
#chr14: start 16000000
#chr15: start 17000000
#chr21: start 5010000
#chr22: start 10500000 
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
startAtSegmentDict ={}

#chromosomeOrderList = [ 'hg18_chr12']
#chromosomeDict = {'hg18_chr12':[0,1e9]}

#chromosomeOrderList = [ 'hg38_chr22']
#chromosomeDict = {'hg38_chr22':[10500000,1e9]}

#lstm11 12/3
#chromosomeOrderList = ['hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
#chromosomeDict = { 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[17e6,1e9], 'hg38_chr4':[16e6,1e9], 'hg38_chr3':[16e6,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
#startAtSegmentDict ={'hg38_chr10':54}

#chromosomeOrderList = ['hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
#chromosomeDict = {'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
#startAtSegmentDict = {'hg38_chr5':159, 'hg38_chr4':0, 'hg38_chr3':0, 'hg38_chr2':0, 'hg38_chr1':0}
#

#lstm4 (5) 12/3
#chromosomeOrderList = [  'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
#chromosomeDict = {  'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
#startAtSegmentDict ={ 'hg38_chr3':75}


#lstm11 4/4
#chromosomeOrderList = [ 'hg38_chr1']
#chromosomeDict = { 'hg38_chr1':[0,1e9]}
#startAtSegmentDict ={ 'hg38_chr1':169}


#chromosomeOrderList = ['hg38_chr13']
#chromosomeDict = {'hg38_chr13':[16000000,1e9]}
#startAtSegmentDict ={'hg38_chr13':45}

#lstm1 23/5, redo of chr3,4,5 since error in startAtSegment's:
#chromosomeOrderList = [  'hg38_chr5', 'hg38_chr4', 'hg38_chr3']
#chromosomeDict = {  'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9]}

#chromosomeOrderList = [ 'hg38_chr5']
#chromosomeDict = { 'hg38_chr5':[0,1e9]}
#startAtSegmentDict ={'hg38_chr3':67}

#lstm1 10/6; redo chr9, seg's 56-59 (hardcoded the nrSegments to be 15)
#chromosomeOrderList = [ 'hg38_chr9']
#chromosomeDict = { 'hg38_chr9':[0,1e9]}
#startAtSegmentDict ={ 'hg38_chr9':46}


#lstm4s
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr16',  'hg38_chr14',  'hg38_chr12', 'hg38_chr10',  'hg38_chr8',  'hg38_chr6',  'hg38_chr4',  'hg38_chr2',  'hg38_chr17',  'hg38_chr15',  'hg38_chr13', 'hg38_chr11', 'hg38_chr9',  'hg38_chr7', 'hg38_chr5', 'hg38_chr3', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

chromosomeOrderList = [ 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
startAtSegmentDict ={'hg38_chr1':203}


#LSTM4P and LSTM4 on part2:
chromosomeOrderList = ['hg38_part2_chr22', 'hg38_part2_chr21', 'hg38_part2_chr20', 'hg38_part2_chr19', 'hg38_part2_chr18', 'hg38_part2_chr17', 'hg38_part2_chr16', 'hg38_part2_chr15', 'hg38_part2_chr14', 'hg38_part2_chr13', 'hg38_part2_chr12', 'hg38_part2_chr11', 'hg38_part2_chr10', 'hg38_part2_chr9', 'hg38_part2_chr8', 'hg38_part2_chr7', 'hg38_part2_chr6', 'hg38_part2_chr5', 'hg38_part2_chr4', 'hg38_part2_chr3', 'hg38_part2_chr2', 'hg38_part2_chr1']
chromosomeDict = {'hg38_part2_chr22':[0,1e9], 'hg38_part2_chr21':[0,1e9], 'hg38_part2_chr20':[0,1e9], 'hg38_part2_chr19':[0,1e9], 'hg38_part2_chr18':[0,1e9], 'hg38_part2_chr17':[0,1e9], 'hg38_part2_chr16':[0,1e9], 'hg38_part2_chr15':[0,1e9], 'hg38_part2_chr14':[0,1e9], 'hg38_part2_chr13':[0,1e9], 'hg38_part2_chr12':[0,1e9], 'hg38_part2_chr11':[0,1e9], 'hg38_part2_chr10':[0,1e9], 'hg38_part2_chr9':[0,1e9], 'hg38_part2_chr8':[0,1e9], 'hg38_part2_chr7':[0,1e9], 'hg38_part2_chr6':[0,1e9], 'hg38_part2_chr5':[0,1e9], 'hg38_part2_chr4':[0,1e9], 'hg38_part2_chr3':[0,1e9], 'hg38_part2_chr2':[0,1e9], 'hg38_part2_chr1':[0,1e9]}
startAtSegmentDict ={}


 
#lstm4s2
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9]}
   

stats.predictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                         startAtSegmentDict = startAtSegmentDict,
                        segmentLength = segmentLength,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b, #!!!!!
                        customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b, #use 1 for bi-directional models
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b)
                    

#To compute the accuracy separately, one may run this (takes that the model's prob distribution of the 
four bases at every position is known --- the predArray --- eg by running predictOnChromosomes as right above):

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9]}

#chromosomeOrderList = [ 'hg38_chr22']
#chromosomeDict = {'hg38_chr22':[10500000,1e9]}
#startAtSegmentDict = {}

chromosomeOrderList = ['hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']
chromosomeDict = {'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9]}
startAtSegmentDict = {'hg38_chr16':45}


#set this so that it matches the setting for the computed predArray(s)
segmentLength = 1000000
averageRevComplementary_b = 0

#set these as desired
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
defaultAccuracy = 0.25

stats.computeAccuracyOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength,
                        startAtSegmentDict = startAtSegmentDict,
                        averageRevComplementary_b = averageRevComplementary_b, #!!!!!
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        defaultAccuracy = defaultAccuracy,
                        on_binf_b = on_binf_b) 
                        

#Mouse

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse mm9 genome:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr200'

augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize = 50
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1000000

batchSize = 528
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1

#mouseLSTM4 on mouse m38:
chromosomeOrderList = [ 'm38_chr1', 'm38_chr2',  'm38_chr4',  'm38_chr6',  'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
chromosomeDict = { 'm38_chr1':[0,1e9],'m38_chr2':[0,1e9],'m38_chr4':[0,1e9],'m38_chr6':[0,1e9],'m38_chr8':[0,1e9],'m38_chr9':[0,1e9],'m38_chr10':[0,1e9],'m38_chr11':[0,1e9],'m38_chr12':[0,1e9],'m38_chr13':[0,1e9],'m38_chr14':[0,1e9],'m38_chr15':[0,1e9],'m38_chr16':[0,1e9],'m38_chr17':[0,1e9],'m38_chr18':[0,1e9],'m38_chr19':[0,1e9]}
startAtSegmentDict ={}

#mouseLSTM4 on mouse mm9:
chromosomeOrderList = [ 'mm9_chr3', 'mm9_chr5', 'mm9_chr7', 'mm9_chr11']
chromosomeDict = { 'mm9_chr3':[0,1e9], 'mm9_chr5':[0,1e9], 'mm9_chr7':[0,1e9], 'mm9_chr11':[0,1e9]}
startAtSegmentDict ={'mm9_chr3':61, 'mm9_chr5':61, 'mm9_chr7':61, 'mm9_chr11':61}


stats.predictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                         startAtSegmentDict = startAtSegmentDict,
                        segmentLength = segmentLength,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b, #!!!!!
                        customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b, #use 1 for bi-directional models
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b)


#Yeast:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"

#train test split, LSTM4
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"

#no train test split, LSTM4
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/noTrainTestSplit/avgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/noTrainTestSplit/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr8"

#train test split, LSTM41 (as LSTM4 but "taken out" much earlier -- at repeat 1!)
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr1"



augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize = 50
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1e5

batchSize = 528
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1

#start positions
chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}
startAtSegmentDict = {}


stats.predictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b, #!!!!!
                        customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b, #use 1 for bi-directional models
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b)
                        

#To compute the accuracy separately, one may run this (takes that the model's prob distribution of the 
four bases at every position is known --- the predArray --- eg by running predictOnChromosomes as right above:

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}


#set this so that it matches the setting for the computed predArray(s)
segmentLength = 1e5
averageRevComplementary_b = 0

#set these as desired
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
defaultAccuracy = 0.25

stats.computeAccuracyOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength, 
                        averageRevComplementary_b = averageRevComplementary_b, #!!!!!
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        defaultAccuracy = defaultAccuracy,
                        on_binf_b = on_binf_b)
                     
 
  
 
#Droso:                       

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/newSplitFct/"

#train test split, LSTM4
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"

#no train test split, LSTM4
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/noTrainTestSplit/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/noTrainTestSplit/"
modelFileNameNN ="modelLSTM_modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr157"


#train test split, LSTM41
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr12"


augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize = 50
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1e6

batchSize = 528
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1

#start positions
chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
chromosomeDict = {'r6.18_chrX':[0,1e8], 'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8], 'r6.18_chr3L':[0,1e8], 'r6.18_chr3R':[0,1e8],'r6.18_chr4':[0,1e8] }
startAtSegmentDict = {}

chromosomeOrderList = [ 'r6.18_chr2L', 'r6.18_chr2R']
chromosomeDict = {'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8]} 
startAtSegmentDict = {}


stats.predictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b, #!!!!!
                        customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b, #use 1 for bi-directional models
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b)


#To compute the accuracy separately, one may run this (takes that the model's prob distribution of the 
four bases at every position is known --- the predArray --- eg by running predictOnChromosomes as right above:

chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
chromosomeDict = {'r6.18_chrX':[0,1e8], 'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8], 'r6.18_chr3L':[0,1e8], 'r6.18_chr3R':[0,1e8],'r6.18_chr4':[0,1e8] }

chromosomeOrderList = ['r6.18_chrX']

#set this so that it matches the setting for the computed predArray(s)
segmentLength = 1e6
averageRevComplementary_b = 0

#set these as desired
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
defaultAccuracy = 0.25

stats.computeAccuracyOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength, 
                        averageRevComplementary_b = averageRevComplementary_b, #!!!!!
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        defaultAccuracy = defaultAccuracy,
                        on_binf_b = on_binf_b)
                     
                        



#Zebrafish

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/trainTestSplit_80_20/notAvgRevCompl/"

rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/trainTestSplit_80_20/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"



augmentWithRevComplementary_b = 0 #!!!!!
leftRight_b = 1
customFlankSize = 50
computePredAcc_b = 1
Fourier_b = 0

segmentLength = 1e6

batchSize = 528
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1

#start positions; in this assembly no chr's have a length above 1e8 and there are no large heading/trailing sections of Ns:
chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr1':[0,1e8], 'GRCz11_chr2':[0,1e8], 'GRCz11_chr3':[0,1e8], 'GRCz11_chr4':[0,1e8], 'GRCz11_chr5':[0,1e8], 'GRCz11_chr6':[0,1e8], 'GRCz11_chr7':[0,1e8], 'GRCz11_chr8':[0,1e8],'GRCz11_chr9':[0,1e8], 'GRCz11_chr10':[0,1e8], 'GRCz11_chr11':[0,1e8], 'GRCz11_chr12':[0,1e8],'GRCz11_chr13':[0,1e8], 'GRCz11_chr14':[0,1e8], 'GRCz11_chr15':[0,1e8], 'GRCz11_chr16':[0,1e8], 'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}

chromosomeOrderList = ['GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}


chromosomeOrderList = ['GRCz11_chr16']
chromosomeDict = {'GRCz11_chr16':[0,1e8]}

startAtSegmentDict = {'GRCz11_chr16':44}

stats.predictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength,
                        startAtSegmentDict = startAtSegmentDict,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b, #!!!!!
                        customFlankSize = customFlankSize,
                        computePredAcc_b = computePredAcc_b, 
                        overlap = 0,
                        leftRight_b = leftRight_b, #use 1 for bi-directional models
                        batchSize = batchSize,
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b)


_________________________________________________________

Secondary use: 
Runs aimed at revealing periodicity in base composition, AT- or GC-rich
These take recoding the labeling of the output, ie the "mid bases"
This is done in the computeAccuracyOnSamples fct 
_________________________________________________________

                    
What happens doing this is that (in AT bias case) if the base at a position is an A or T, the 
model's prediction will always be true no matter what it has predicted, and always wrong
if the position happened to be a C or G.
 
#Human
 
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"


#LSTM4/5, flanks 50, trained on hg38; GC bias only uses the q-arrays, and we use the flanks 50 version then and not the 200: 
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
 
rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38"  

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
startAtSegmentDict ={}


#LSTM4 on hg18 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
#GC bias only uses the q-arrays, and we use the flanks 50 version then and not the 200: 
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18"  

chromosomeOrderList = ['hg18_chr12']
chromosomeDict = {'hg18_chr12':[0,1e9]}
startAtSegmentDict ={}




#mouse, m38:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/" 

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38"  

chromosomeOrderList = [ 'm38_chr3', 'm38_chr5', 'm38_chr7', 'm38_chr11']
chromosomeDict = { 'm38_chr3':[0,1e9], 'm38_chr5':[0,1e9], 'm38_chr7':[0,1e9], 'm38_chr11':[0,1e9]}
startAtSegmentDict ={}

chromosomeOrderList = [ 'm38_chr1', 'm38_chr2',  'm38_chr4',  'm38_chr6',  'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
chromosomeDict = { 'm38_chr1':[0,1e9],'m38_chr2':[0,1e9],'m38_chr4':[0,1e9],'m38_chr6':[0,1e9],'m38_chr8':[0,1e9],'m38_chr9':[0,1e9],'m38_chr10':[0,1e9],'m38_chr11':[0,1e9],'m38_chr12':[0,1e9],'m38_chr13':[0,1e9],'m38_chr14':[0,1e9],'m38_chr15':[0,1e9],'m38_chr16':[0,1e9],'m38_chr17':[0,1e9],'m38_chr18':[0,1e9],'m38_chr19':[0,1e9]}
startAtSegmentDict ={}



#mouse, mm9:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/" 

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse mm9 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr200'

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9"  

chromosomeOrderList = ['mm9_chr11', 'mm9_chr7','mm9_chr5', 'mm9_chr3']
chromosomeDict = { 'mm9_chr3':[0,1e9], 'mm9_chr5':[0,1e9], 'mm9_chr7':[0,1e9], 'mm9_chr11':[0,1e9]}
startAtSegmentDict ={}


#Yeast:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"

#train test split
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64"  

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}
startAtSegmentDict = {}

#Droso:                       

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/newSplitFct/"

#train test split
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18"  

chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
chromosomeDict = {'r6.18_chrX':[0,1e8], 'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8], 'r6.18_chr3L':[0,1e8], 'r6.18_chr3R':[0,1e8],'r6.18_chr4':[0,1e8] }
startAtSegmentDict = {}




#Zebrafish

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"

rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11"  

chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr1':[0,1e8], 'GRCz11_chr2':[0,1e8], 'GRCz11_chr3':[0,1e8], 'GRCz11_chr4':[0,1e8], 'GRCz11_chr5':[0,1e8], 'GRCz11_chr6':[0,1e8], 'GRCz11_chr7':[0,1e8], 'GRCz11_chr8':[0,1e8],'GRCz11_chr9':[0,1e8], 'GRCz11_chr10':[0,1e8], 'GRCz11_chr11':[0,1e8], 'GRCz11_chr12':[0,1e8],'GRCz11_chr13':[0,1e8], 'GRCz11_chr14':[0,1e8], 'GRCz11_chr15':[0,1e8], 'GRCz11_chr16':[0,1e8], 'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}
startAtSegmentDict = {}


chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr4','GRCz11_chr6',  'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10',  'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr1':[0,1e8], 'GRCz11_chr2':[0,1e8],'GRCz11_chr4':[0,1e8],  'GRCz11_chr6':[0,1e8],  'GRCz11_chr8':[0,1e8],'GRCz11_chr9':[0,1e8], 'GRCz11_chr10':[0,1e8],  'GRCz11_chr12':[0,1e8],'GRCz11_chr13':[0,1e8], 'GRCz11_chr14':[0,1e8], 'GRCz11_chr15':[0,1e8], 'GRCz11_chr16':[0,1e8], 'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}
startAtSegmentDict = {}


#Human on alternative sequence: direct download of chromo seq's from UCSC:
 
rootGenome = r"/binf-isilon/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/checkChromoSeqs/"

#LSTM4/5, flanks 50, trained on hg38; GC bias only uses the q-arrays, and we use the flanks 50 version then and not the 200: 
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
 
rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38_ucscDirect"  
  
chromosomeOrderList = ['hg38_chr19', 'hg38_chr17',  'hg38_chr5']
chromosomeDict = {'hg38_chr19':[0,1e9], 'hg38_chr17':[0,1e9],  'hg38_chr5':[0,1e9]}
startAtSegmentDict ={}
extension = '.fa'



#set this so that it matches the setting for the computed predArray(s)
segmentLength = 1000000
averageRevComplementary_b = 0

#set these as desired
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
defaultAccuracy = 0.25

#Use one (!) of these two:

forATorGCbias_b = 1 #!!!!!!!!!!!!
#For AT bias:
recodeA = [1,1,1,1]
recodeC = [0,0,0,0]
recodeG = [0,0,0,0]
recodeT = [1,1,1,1]
modelFileName_forATorGCbias = 'ATbias'
rootOutput_forATorGCbias  = rootOutputBias + r"/ATbias/notAvgRevCompl/"


#CG bias
forATorGCbias_b = 1 #!!!!!!!!!!!!
recodeA = [0,0,0,0]
recodeC = [1,1,1,1]
recodeG = [1,1,1,1]
recodeT = [0,0,0,0] 
modelFileName_forATorGCbias ="GCbias"
rootOutput_forATorGCbias  = rootOutputBias + r"/GCbias/notAvgRevCompl/"


stats.computeAccuracyOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileNameNN,
                        segmentLength = segmentLength,
                        startAtSegmentDict = startAtSegmentDict,
                        averageRevComplementary_b = averageRevComplementary_b, #!!!!!
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        defaultAccuracy = defaultAccuracy,
                        on_binf_b = on_binf_b,
                        forATorGCbias_b = forATorGCbias_b, 
                        rootOutput_forATorGCbias= rootOutput_forATorGCbias,
                        modelFileName_forATorGCbias = modelFileName_forATorGCbias,
                        recodeA = recodeA,
                             recodeC = recodeC,
                             recodeG = recodeG,
                             recodeT = recodeT
#                             ,extension = extension
                             
                        )
                        
                        

#######################################################################

# Part 4.  Get predictions from external model across list of chromo's

#######################################################################

      
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"

chromosomeOrderList = ['hg38_chr22'] #, 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']
chromosomeDict = {'hg38_chr22':[10500000,1e9]} #, 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9]}


#Markov model:
rootOutput  = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/Markovs/Markov14/avgRevCompl/"
rootModel = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/Markovs/Markov14/"
modelFileName = 'Bidirectional_Markov_win14'              
            
modelPredicitionFileNameDict = {'hg38_chr22':r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'}
averageRevComplementary_b = 1 #!!!
      
displacementOfExternal = 1
kModel= 14
kMer_b = 0


#k-mer model

rootOutput = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/'

#place holders
modelPredicitionFileNameDict = {}
averageRevComplementary_b = 0

kMer_b = 1
kModel = 5
modelFileName = '5-mer'



#Set these as desired (indep of model)

segmentLength = 1000000
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
       
       
stats.externalModelPredictOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         modelFileName = modelFileName,
                         modelPredicitionFileNameDict = modelPredicitionFileNameDict, 
                         averageRevComplementary_b = averageRevComplementary_b,
                         displacementOfExternal = displacementOfExternal, 
                         kModel = kModel,
                         kMer_b = kMer_b,
                        segmentLength = segmentLength,
#                        startAtSegmentDict = {}, 
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        on_binf_b = on_binf_b
                        )        
                        


                     
#set this so that it matches the setting for the computed predArray(s)
segmentLength = 1000000
averageRevComplementary_b = 1

#set these as desired
windowLength = 1
stepSize = 1
Fourier_b = 0
on_binf_b = 1
defaultAccuracy = 0.25

stats.computeAccuracyOnChromosomes(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                         chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         rootModel = rootModel,
                         modelFileName = modelFileName,
                        segmentLength = segmentLength,
#                        startAtSegmentDict = startAtSegmentDict,
                        averageRevComplementary_b = averageRevComplementary_b, #!!!!!
                        windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b,
                        defaultAccuracy = defaultAccuracy,
                        on_binf_b = on_binf_b)
        
                

######################################################################################3

# Part 5. "Big scale" LR tests and plotting for comparison of two models.

######################################################################################3

#To start with a small genome ... just to get past all the small-errors correcting wo having to wait endlessly ...
#---------------
#Yeast genome 
#---------------

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"
fileGenome = rootGenome +fileName

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}

chromosomeOrderList = ['R64_chr1']
chromosomeDict = {'R64_chr1':[0,1e8]}


#LR test frq vs LSTM1
#LSTM
modelName2 = "LSTM1"
modelType2 = 0
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/noTrainTestSplit/avgRevCompl/"
modelFile2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr8"
augmentWithRevComplementary2_b = 1
flankSize2 = 200
segmentLength2 = 100000


#Frq model
modelName1 = "4mer"
modelType1 = 1
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/GRCh38.p12/"
modelFile1 = '' #placeholder
flankSize1 = 4
augmentWithRevComplementary1_b = 0 #placeholder
segmentLength1 = 1000000 #placeholder


#Another try, here yeast vs yeast (as it should be)

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"
fileGenome = rootGenome +fileName

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8]}


#LR test frq vs LSTM1
#LSTM
modelName2 = "LSTM1"
modelType2 = 0
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/noTrainTestSplit/avgRevCompl/"
modelFile2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr8"
augmentWithRevComplementary2_b = 1
flankSize2 = 200
segmentLength2 = 100000


#LSTM2 as nr2:
modelName1 = "LSTM2"
modelType1 = 0
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/trainTestSplit_80_20/notAvgRevCompl/"
modelFile1 ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"
augmentWithRevComplementary1_b = 0
flankSize1 = 50
segmentLength1 = 100000



rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/noTrainTestSplit/avgRevCompl/"

#just try it out -- these annotatios are for the human genome!
rootAnnotationFiles = r'/binf-isilon/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/'
annotationTypes = ['simpleRepeats', 'repeat', 'cds', 'introns', '3UTR', '5UTR', 'gene']   
   

LRtest_b = 0
plot_b = 1
bins = 50
chromoFieldIdx = 4
useSubSampleLR_b = 1 
subSampleSizeLR = 10000000
subSampleFractionLR = 1 #no effect when subSampleSizeLR set > 0
stats.compareTwoModelsOnChromosomes(rootGenome = rootGenome, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, rootOutput =rootOutput, rootPredictModel1 = rootPredictModel1,  modelFile1 = modelFile1, flankSize1 = flankSize1 , modelType1 = modelType1, modelName1 = modelName1, rootPredictModel2 = rootPredictModel2,  modelFile2 = modelFile2, flankSize2 = flankSize2, modelType2 = modelType2, modelName2 = modelName2, averageRevComplementary2_b = augmentWithRevComplementary2_b, segmentLength2 = segmentLength2, averageRevComplementary1_b = augmentWithRevComplementary1_b, segmentLength1 = segmentLength1, LRtest_b = LRtest_b, useSubSampleLR_b = useSubSampleLR_b, subSampleSizeLR = subSampleSizeLR, subSampleFractionLR = subSampleFractionLR, plot_b = plot_b, bins = bins, chromoFieldIdx = chromoFieldIdx, rootAnnotationFiles = rootAnnotationFiles, annotationTypes = annotationTypes)




#---------------
#Human
#---------------

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"

#LR test frq vs LSTM1
#LSTM1 w/wo avg on rev compl:
modelName2 = "LSTM1"
modelType2 = 0
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFile2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
augmentWithRevComplementary2_b = 0
flankSize2 = 200
segmentLength2 = 1000000

#Frq model
modelName1 = "4mer"
modelType1 = 1
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/"
modelFile1 = '' #placeholder
flankSize1 = 4
augmentWithRevComplementary1_b = 0 #placeholder
segmentLength1 = 1000000 #placeholder




#Frq model vs LSTM1 (model 1 and 2 swapped compared to above)
#LSTM1 w/wo avg on rev compl:
modelName1 = "LSTM1"
modelType1 = 0
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFile1 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
augmentWithRevComplementary1_b = 0
flankSize1 = 200
segmentLength1 = 1000000

#Frq model
modelName2 = "4mer"
modelType2 = 1
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/"
modelFile2 = '' #placeholder
flankSize2 = 4
augmentWithRevComplementary2_b = 0 #placeholder
segmentLength2 = 1000000 #placeholder

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"


################################
#Markov k14 vs LSTM1 (on hg38)


modelName1 = "Markov k=14"
modelType1 = 2
rootPredictModel1 = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/03.Bidir_Markov_model/"
modelFile1 = '' #placeholder
flankSize1 = 14
augmentWithRevComplementary1_b = 0 #placeholder
segmentLength1 = 1000000 #placeholder


modelName2 = "LSTM1"
modelType2 = 0
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFile2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
augmentWithRevComplementary2_b = 0
flankSize2 = 200
segmentLength2 = 1000000

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"



################################
#LSTM4 vs LSTM1 (on hg38) : 
USE compareOneModelToListOfModelsOnChromosomes for this case (modelType1 = 2 is not supported in compareTwoModelsOnChromosomes)

#modelName1 = "LSTM4"
#modelType1 = 0
#rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
#modelFile1 = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#augmentWithRevComplementary1_b = 0
#flankSize1 = 50
#segmentLength1 = 1000000
#
#
#
#modelName2 = "LSTM1"
#modelType2 = 0
#rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
#modelFile2 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
#augmentWithRevComplementary2_b = 0
#flankSize2 = 200
#segmentLength2 = 1000000
#
#rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"


################################
#LSTM4 vs mouseLSTM4 (on hg38)
modelName1 = "LSTM4"
modelType1 = 0
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFile1 = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
augmentWithRevComplementary1_b = 0
flankSize1 = 50
segmentLength1 = 1000000


#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
modelName2 = "mouseLSTM4"
modelType2 = 0
rootPredictModel2 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFile2 = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
augmentWithRevComplementary2_b = 0
flankSize2 = 50
segmentLength2 = 1000000

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"

################################

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
annotationTypes = ['simple_repeat', 'repeat', 'RmskAndTrf', 'cds', 'introns', '3UTR', '5UTR', 'gene', 'repeatsGenomeSeq']   
#annotationTypes = ['simple_repeat']   

#chromosomeOrderList = ['hg38_chr19', 'hg38_chr18', 'hg38_chr17','hg38_chr1']
#chromosomeDict = {'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9],'hg38_chr1':[0,1e9]}

chromosomeOrderList = ['hg38_chr1']
chromosomeDict = {'hg38_chr1':[0,1e9]}


nameChangeDict = {'simple_repeat': 'TRF repeats', 'repeat':'Rmsk repeats', 'RmskAndTrf':'Rmsk and trf repeats', 'repeatsGenomeSeq':'Wm repeats'}

#fct call:
LRtest_b = 0
plot_b = 1
bins = 50
chromoFieldIdx = 4
useSubSampleLR_b = 1 
subSampleSizeLR = 0 #10000000
subSampleFractionLR = 0.1 #no effect when subSampleSizeLR set > 0
saveAtDpi = 300
stats.compareTwoModelsOnChromosomes(rootGenome = rootGenome, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, rootOutput =rootOutput, rootPredictModel1 = rootPredictModel1, modelFile1 = modelFile1, flankSize1 = flankSize1 , modelType1 = modelType1, modelName1 = modelName1, rootPredictModel2 = rootPredictModel2, modelFile2 = modelFile2, flankSize2 = flankSize2, modelType2 = modelType2, modelName2 = modelName2, averageRevComplementary2_b = augmentWithRevComplementary2_b, segmentLength2 = segmentLength2, averageRevComplementary1_b = augmentWithRevComplementary1_b, segmentLength1 = segmentLength1, LRtest_b = LRtest_b, useSubSampleLR_b = useSubSampleLR_b, subSampleSizeLR = subSampleSizeLR, subSampleFractionLR = subSampleFractionLR, plot_b = plot_b, bins = bins, chromoFieldIdx = chromoFieldIdx, rootAnnotationFiles = rootAnnotationFiles, annotationTypes = annotationTypes, nameChangeDict = nameChangeDict, saveAtDpi = saveAtDpi)
   
   
##########################################################################################   
## Here a call to compareOneModelToListOfModelsOnChromosomes, which amounts to running 
## compareTwoModelsOnChromosomes for one NN vs a list of simpler models;
## The roles of model1 and model2 are though interchanged!! (in compareTwoModelsOnChromosomes
## model1 is the "base", here it's the other way around):
##########################################################################################
   
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"

#LR test frq vs LSTM1

#LSTM1 w/wo avg on rev compl:
modelName1 = "LSTM1"
modelType1 = 0
rootPredictModel1 = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFile1 ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
augmentWithRevComplementary1_b = 0
flankSize1 = 200
segmentLength1 = 1000000


#List of simpler models: Frq models and Markov

#Lists to hold the models' data:
rootPredictModel2List = []
modelFile2List = []
flankSize2List = []
modelType2List = []
modelName2List = []
averageRevComplementary2List_b = []
segmentLength2List = [] 

#Populate the lists:

rootPredictModel2 = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/"
modelName2 = "k=3 central"
modelName2 = "3mer"
modelType2 = 1
modelFile2 = '' #placeholder
flankSize2 = 3
augmentWithRevComplementary2_b = 0 #placeholder
segmentLength2 = 1000000 #placeholder

rootPredictModel2List.append(rootPredictModel2)
modelFile2List.append(modelFile2)
flankSize2List.append(flankSize2)
modelType2List.append(modelType2)
modelName2List.append(modelName2)
averageRevComplementary2List_b.append(augmentWithRevComplementary2_b)
segmentLength2List.append(segmentLength2)


rootPredictModel2 = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/"
modelName2 = "k=4 central"
modelName2 = "4mer"
modelType2 = 1
modelFile2 = '' #placeholder
flankSize2 = 4
augmentWithRevComplementary2_b = 0 #placeholder
segmentLength2 = 1000000 #placeholder

rootPredictModel2List.append(rootPredictModel2)
modelFile2List.append(modelFile2)
flankSize2List.append(flankSize2)
modelType2List.append(modelType2)
modelName2List.append(modelName2)
averageRevComplementary2List_b.append(augmentWithRevComplementary2_b)
segmentLength2List.append(segmentLength2)


rootPredictModel2 = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/"
modelName2 = "k=5 central"
modelName2 = "5mer"
modelType2 = 1
modelFile2 = '' #placeholder
flankSize2 = 5
augmentWithRevComplementary2_b = 0 #placeholder
segmentLength2 = 1000000 #placeholder

rootPredictModel2List.append(rootPredictModel2)
modelFile2List.append(modelFile2)
flankSize2List.append(flankSize2)
modelType2List.append(modelType2)
modelName2List.append(modelName2)
averageRevComplementary2List_b.append(augmentWithRevComplementary2_b)
segmentLength2List.append(segmentLength2)


rootPredictModel2 = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/03.Bidir_Markov_model/"
modelName2 = "Markov k=14"
modelType2 = 2
modelFile2 = '' #placeholder
flankSize2 = 14
augmentWithRevComplementary2_b = 0 #placeholder
segmentLength2 = 1000000 #placeholder

rootPredictModel2List.append(rootPredictModel2)
modelFile2List.append(modelFile2)
flankSize2List.append(flankSize2)
modelType2List.append(modelType2)
modelName2List.append(modelName2)
averageRevComplementary2List_b.append(augmentWithRevComplementary2_b)
segmentLength2List.append(segmentLength2)


rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"

rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
annotationTypes = ['simple_repeat', 'repeat', 'RmskAndTrf', 'cds', 'introns', '3UTR', '5UTR', 'gene', 'repeatsGenomeSeq']   

nameChangeDict = {'simple_repeat': 'TRF repeats', 'repeat':'Rmsk repeats', 'RmskAndTrf':'Rmsk and trf repeats', 'repeatsGenomeSeq':'Wm repeats'}

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
#chromosomeOrderList = chromosomeOrderList[:-1] #want it to start from chr 1
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

firstChromoNr = 22

chromosomeOrderList = ['hg38_chr22']
chromosomeDict = {'hg38_chr22':[10500000,1e9]}

firstChromoNr = 22

chromosomeOrderList = [     'hg38_chr19', 'hg38_chr18', 'hg38_chr17' ]
#chromosomeOrderList = chromosomeOrderList[:-1] #want it to start from chr 1
chromosomeDict = {'hg38_chr17':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr19':[0,1e9]}

firstChromoNr = 19

chromosomeOrderList = ['hg38_chr1']
chromosomeDict = {'hg38_chr1':[0,1e9]}

firstChromoNr = 1


#fct call:
LRtest_b = 0 
plot_b = 1
bins = 50
chromoFieldIdx = 4
useSubSampleLR_b = 1 
subSampleSizeLR = 0 #10000000
subSampleFractionLR = 0.1 #no effect when subSampleSizeLR set > 0
saveAtDpi = 300


#Call
stats.compareOneModelToListOfModelsOnChromosomes(rootGenome = rootGenome, 
chromosomeOrderList = chromosomeOrderList, 
firstChromoNr = firstChromoNr,
chromosomeDict = chromosomeDict, 
rootOutput =rootOutput, 
rootPredictModel1 = rootPredictModel1, 
modelFile1 = modelFile1, 
flankSize1 = flankSize1 , 
modelType1 = modelType1, 
modelName1 = modelName1, 
rootPredictModel2List = rootPredictModel2List, 
modelFile2List = modelFile2List, 
flankSize2List = flankSize2List, 
modelType2List = modelType2List, 
modelName2List = modelName2List, 
averageRevComplementary2List_b = averageRevComplementary2List_b, 
segmentLength2List = segmentLength2List, 
averageRevComplementary1_b = augmentWithRevComplementary1_b, 
segmentLength1 = segmentLength1, 
LRtest_b = LRtest_b, 
useSubSampleLR_b = useSubSampleLR_b, 
subSampleSizeLR = subSampleSizeLR, 
subSampleFractionLR = subSampleFractionLR, 
plot_b = plot_b, 
bins = bins, 
chromoFieldIdx = chromoFieldIdx, 
rootAnnotationFiles = rootAnnotationFiles, 
annotationTypes = annotationTypes, 
nameChangeDict = nameChangeDict,
saveAtDpi = saveAtDpi)
 
 
---------------------------------------
-- Make tex table and plot of multi-LR 
---------------------------------------
rootResults = rootOutput 
#set ABOVE:
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
modelName1 = modelName1
averageRevComplementary1_b = augmentWithRevComplementary1_b 
modelName2List = modelName2List  
averageRevComplementary2_b_list = averageRevComplementary2List_b
useSubSample_b =  useSubSampleLR_b
subSampleSize = subSampleSizeLR 
subSampleFraction = subSampleFractionLR
rootOutput = rootOutput 
fileName = modelName1 + '_LRtestSimpleModels.txt'
captionText = 'Results of likelihood ratio tests. Test value is the value of the test statistic and Std dev is the standard deviation of it. Model names are as defined in ....'
stats.collectLikelihoodRatioResultsOnChromos(rootResults, chromosomeOrderList, modelName1, averageRevComplementary1_b =averageRevComplementary1_b, modelName2List = modelName2List, averageRevComplementary2_b_list = averageRevComplementary2_b_list, useSubSample_b = useSubSample_b, subSampleSize = subSampleSize, subSampleFraction = subSampleFraction, rootOutput = rootOutput, fileName = fileName, captionText = captionText )
    


######################################################################################3

# Part 6. Accuracy runs

######################################################################################3

_________________________________________________________


Step 1. By segments on single chromo. In Step 2 we run this accross chromos
_________________________________________________________


rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/LSTM1/"

#LSTM1
modelFileName ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/" + chrName + "/"

averageRevComplementary_b = 0

segmentLength = 1000000



chrName = 'hg38_chr22'
genomeIdName = chrName + '_seg' + str(int(segmentLength))


#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1

chromosomeOrderList = [ 'hg38_chr22']
chromosomeDict = {'hg38_chr22':[10500000,1e9]}

startAtPosition, endAtPosition = chromosomeDict[chrName]

#if annotation arrays not read in go to "# One-off's: handling annotation files (bed files); loading etc" below
#and do it (if the bed-files exist, of course, else you must download them first ..)

#load annotationArray if there:
rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
annotationTypes = ['simple_repeat', 'repeat', 'cds', 'introns', '3UTR', '5UTR']
#annotationTypes = ['simple_repeat']
annotationDict = {}
import cPickle as pickle
for annoType in annotationTypes:
    annotationFile = rootAnnotationFiles + chrName + '_annotationArray_' + annoType
    print annotationFile
    annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))
                 
for anno in annotationDict:
    print anno, annotationDict[annoType].shape, np.where(annotationDict[annoType]   == 1)


#
#15531878 15532114
#15532810 15532851
#15532852 15532875
#15533046 15533416
#15533416 15533922
#annotationDict[annoType][15533918] 
#annotationDict[annoType][15536437]
#15536437 15537903
#15537908 15538030

             
resultsDict, resultsDictAggr = stats.getAccuracyOnSegments(rootOutput = rootOutput,
                             modelFileName = modelFileName, 
                             segmentLength = segmentLength,
                             genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                             averageRevComplementary_b = averageRevComplementary_b,                              
                             windowLength = windowLength,
                             stepSize = stepSize,
                             annotationDict = annotationDict,
                             startAtPosition = startAtPosition
                             )


_________________________________________________________     

Step 2. Accross several chromos 
_________________________________________________________    
    

#LSTM1
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileName ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"

#LSTM11: as LSTM1 but at earlier training stage:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileName ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr15"


#LSTM4S; trained on all odd numbered hg38-chromos; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr184'


#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'

#LSTM4P on part2 (test part) of the 80/20-split genome (GRCh38):
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/split/part2/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199'


#GC/AT bias:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'

modelFileName_predReturn = 'GCbias'
rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38"  
rootOutput_predReturn  = rootOutputBias + r"/GCbias/notAvgRevCompl/"


#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'


#Mouse
#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'


#Zebrafish
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/"
modelFileName ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"


#Droso:                       
#w. train test split
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/"
modelFileName ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"

#less trained train test split, LSTM41
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/"
modelFileName ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr12"

#no train test split
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/noTrainTestSplit/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/noTrainTestSplit/"
modelFileName ="modelLSTM_modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr157"


#Yeast, w. train test split, LSTM4
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/"
modelFileName = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"

#Yeast, less trained, w. train test split, LSTM41:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/"
modelFileName = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr1"



#averageRev ..., window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
averageRevComplementary_b = 0

#Human, mouse, droso, zebrafish:
segmentLength = 1000000

#Yeast:
segmentLength = 100000

#For human, hg38
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

#chromosomeOrderList = ['hg38_chr9']
#chromosomeDict = {'hg38_chr9':[0,1e9]}
      
rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
annotationTypes = ['repeatsGenomeSeq', 'simple_repeat', 'repeat', 'cds', 'introns', '3UTR', '5UTR', 'gene']   

#For human hg_38, but split -- ie for LSTM4P
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/split/part2/"
chromosomeOrderList = ['hg38_part2_chr22', 'hg38_part2_chr21', 'hg38_part2_chr20', 'hg38_part2_chr19', 'hg38_part2_chr18', 'hg38_part2_chr17', 'hg38_part2_chr16', 'hg38_part2_chr15', 'hg38_part2_chr14', 'hg38_part2_chr13', 'hg38_part2_chr12', 'hg38_part2_chr11', 'hg38_part2_chr10', 'hg38_part2_chr9', 'hg38_part2_chr8', 'hg38_part2_chr7', 'hg38_part2_chr6', 'hg38_part2_chr5', 'hg38_part2_chr4', 'hg38_part2_chr3', 'hg38_part2_chr2', 'hg38_part2_chr1']
chromosomeDict = {'hg38_part2_chr22':[0,1e9], 'hg38_part2_chr21':[0,1e9], 'hg38_part2_chr20':[0,1e9], 'hg38_part2_chr19':[0,1e9], 'hg38_part2_chr18':[0,1e9], 'hg38_part2_chr17':[0,1e9], 'hg38_part2_chr16':[0,1e9], 'hg38_part2_chr15':[0,1e9], 'hg38_part2_chr14':[0,1e9], 'hg38_part2_chr13':[0,1e9], 'hg38_part2_chr12':[0,1e9], 'hg38_part2_chr11':[0,1e9], 'hg38_part2_chr10':[0,1e9], 'hg38_part2_chr9':[0,1e9], 'hg38_part2_chr8':[0,1e9], 'hg38_part2_chr7':[0,1e9], 'hg38_part2_chr6':[0,1e9], 'hg38_part2_chr5':[0,1e9], 'hg38_part2_chr4':[0,1e9], 'hg38_part2_chr3':[0,1e9], 'hg38_part2_chr2':[0,1e9], 'hg38_part2_chr1':[0,1e9]}
startAtSegmentDict ={}
rootAnnotationFiles = ''
annotationTypes = []   




#For mouseLSTM4 on mouse m38:
chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
chromosomeDict = { 'm38_chr1':[0,1e9],'m38_chr2':[0,1e9],'m38_chr3':[0,1e9], 'm38_chr4':[0,1e9], 'm38_chr5':[0,1e9], 'm38_chr6':[0,1e9],'m38_chr7':[0,1e9],'m38_chr8':[0,1e9],'m38_chr9':[0,1e9],'m38_chr10':[0,1e9],'m38_chr11':[0,1e9],'m38_chr12':[0,1e9],'m38_chr13':[0,1e9],'m38_chr14':[0,1e9],'m38_chr15':[0,1e9],'m38_chr16':[0,1e9],'m38_chr17':[0,1e9],'m38_chr18':[0,1e9],'m38_chr19':[0,1e9]}

rootAnnotationFiles = '/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/'
annotationTypes = ['repeatsGenomeSeq']

#For zebrafish
chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr1':[0,1e8], 'GRCz11_chr2':[0,1e8], 'GRCz11_chr3':[0,1e8], 'GRCz11_chr4':[0,1e8], 'GRCz11_chr5':[0,1e8], 'GRCz11_chr6':[0,1e8], 'GRCz11_chr7':[0,1e8], 'GRCz11_chr8':[0,1e8],'GRCz11_chr9':[0,1e8], 'GRCz11_chr10':[0,1e8], 'GRCz11_chr11':[0,1e8], 'GRCz11_chr12':[0,1e8],'GRCz11_chr13':[0,1e8], 'GRCz11_chr14':[0,1e8], 'GRCz11_chr15':[0,1e8], 'GRCz11_chr16':[0,1e8], 'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
annotationTypes = ['repeatsGenomeSeq']

#For droso:
chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
chromosomeDict = {'r6.18_chrX':[0,1e8], 'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8], 'r6.18_chr3L':[0,1e8], 'r6.18_chr3R':[0,1e8],'r6.18_chr4':[0,1e8] }
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/"
annotationTypes = []

#For yeast:
chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
annotationTypes = ['simpleRepeats']




#.. and run it
resultsDictByAnnoSeg, resultsDictByAnno  = stats.getAccuracyChromosomes(chromosomeOrderList = chromosomeOrderList, 
                         rootOutput = rootOutput,
                         modelFileName = modelFileName, 
                         segmentLength = segmentLength,
                         averageRevComplementary_b = averageRevComplementary_b,
                         windowLength = windowLength,
                         stepSize = stepSize, 
                         annotationTypes = annotationTypes,
                         rootAnnotationFiles = rootAnnotationFiles,
                         chromosomeDict = chromosomeDict)
#Only for GC/AT:
#                         ,                         rootOutput_predReturn = rootOutput_predReturn,
#                         modelFileName_predReturn = modelFileName_predReturn
#                         )
    

#Do the aggregation so that it covers all chromosomes/anotations for which results are had:
dictionaryName = 'accuracyChromoByAnnoDictionary'
#human
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}
#human, part2 (LSTM4P)
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/split/part2/"
chromosomeDict = {'hg38_part2_chr22':[0,1e9], 'hg38_part2_chr21':[0,1e9], 'hg38_part2_chr20':[0,1e9], 'hg38_part2_chr19':[0,1e9], 'hg38_part2_chr18':[0,1e9], 'hg38_part2_chr17':[0,1e9], 'hg38_part2_chr16':[0,1e9], 'hg38_part2_chr15':[0,1e9], 'hg38_part2_chr14':[0,1e9], 'hg38_part2_chr13':[0,1e9], 'hg38_part2_chr12':[0,1e9], 'hg38_part2_chr11':[0,1e9], 'hg38_part2_chr10':[0,1e9], 'hg38_part2_chr9':[0,1e9], 'hg38_part2_chr8':[0,1e9], 'hg38_part2_chr7':[0,1e9], 'hg38_part2_chr6':[0,1e9], 'hg38_part2_chr5':[0,1e9], 'hg38_part2_chr4':[0,1e9], 'hg38_part2_chr3':[0,1e9], 'hg38_part2_chr2':[0,1e9], 'hg38_part2_chr1':[0,1e9]}
#mouse
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
chromosomeDict = { 'm38_chr1':[0,1e9],'m38_chr2':[0,1e9],'m38_chr3':[0,1e9], 'm38_chr4':[0,1e9], 'm38_chr5':[0,1e9], 'm38_chr6':[0,1e9],'m38_chr7':[0,1e9],'m38_chr8':[0,1e9],'m38_chr9':[0,1e9],'m38_chr10':[0,1e9],'m38_chr11':[0,1e9],'m38_chr12':[0,1e9],'m38_chr13':[0,1e9],'m38_chr14':[0,1e9],'m38_chr15':[0,1e9],'m38_chr16':[0,1e9],'m38_chr17':[0,1e9],'m38_chr18':[0,1e9],'m38_chr19':[0,1e9]}
#zebrafish
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
chromosomeDict = {'GRCz11_chr1':[0,1e8], 'GRCz11_chr2':[0,1e8], 'GRCz11_chr3':[0,1e8], 'GRCz11_chr4':[0,1e8], 'GRCz11_chr5':[0,1e8], 'GRCz11_chr6':[0,1e8], 'GRCz11_chr7':[0,1e8], 'GRCz11_chr8':[0,1e8],'GRCz11_chr9':[0,1e8], 'GRCz11_chr10':[0,1e8], 'GRCz11_chr11':[0,1e8], 'GRCz11_chr12':[0,1e8],'GRCz11_chr13':[0,1e8], 'GRCz11_chr14':[0,1e8], 'GRCz11_chr15':[0,1e8], 'GRCz11_chr16':[0,1e8], 'GRCz11_chr17':[0,1e8], 'GRCz11_chr18':[0,1e8],'GRCz11_chr19':[0,1e8], 'GRCz11_chr20':[0,1e8], 'GRCz11_chr21':[0,1e8], 'GRCz11_chr22':[0,1e8],'GRCz11_chr23':[0,1e8], 'GRCz11_chr24':[0,1e8], 'GRCz11_chr25':[0,1e8]}
#droso
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/newSplitFct/"
chromosomeDict = {'r6.18_chrX':[0,1e8], 'r6.18_chr2L':[0,1e8], 'r6.18_chr2R':[0,1e8], 'r6.18_chr3L':[0,1e8], 'r6.18_chr3R':[0,1e8],'r6.18_chr4':[0,1e8] }
#yeast
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}


stats.calculateAggrAccOnChromos(rootOutput =rootOutput , chromosomeDict= chromosomeDict,  dictionaryName = dictionaryName)


#The same, but for GC/AT content:
stats.calculateAggrAccOnChromos(rootOutput =rootOutput_predReturn , chromosomeDict= chromosomeDict,  dictionaryName = dictionaryName)


#This returns a dict holding stats on the input to calculateAggrAccOnChromos -- size of qual, annotated part ..: 
rootOutput = rootOutput
rootGenome = rootGenome
#chromosomeOrderList = chromosomeOrderList[::-1]
annotationTypes = annotationTypes 
inclChromoLengthInTable_b = 0
#captionText = 'Human, hg38. Statistics on input to the prediction, LSTM11.'
#fileName = 'table_chromoStats_GRCh38_LSTM11.txt'
captionText = 'Human, hg38. Statistics on input to the prediction, LSTM50P.'
fileName = 'table_chromoStats_GRCh38_LSTM4P.txt'
#captionText = 'Fruit fly, r6.18. Statistics on input to the prediction (LSTM41).'
#fileName = 'table_chromoStats_r6.18_LSTM4.txt'
#captionText = 'Yeast, R64. Statistics on input to the prediction (LSTM41).'
#fileName = 'table_chromoStats_R64.txt'
#captionText = 'Mouse genome, assembly GRCm38 (mm10). Statistics on input to the prediction.'
#fileName = 'table_chromoStats_GRCm38.txt'
outputDict = stats.getInfo_calculateAggrAccOnChromos(rootOutput =rootOutput , chromosomeOrderList = chromosomeOrderList, chromosomeDict= chromosomeDict, annotationList = annotationTypes, rootGenome = rootGenome, dictionaryNameAggr = 'accuracyByChromoAnnoDictionary', dictionaryNameSeg = 'accuracyByChromoAnnoSegDictionary', inclChromoLengthInTable_b = inclChromoLengthInTable_b, captionText = captionText, fileName = fileName)


-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
Step2b. Aggregate accuracy odd/even chromosomes
-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

#For human, hg38
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

#Odd/even partition:
part1 = ['hg38_chr21', 'hg38_chr19', 'hg38_chr17', 'hg38_chr15', 'hg38_chr13', 'hg38_chr11', 'hg38_chr9', 'hg38_chr7', 'hg38_chr5', 'hg38_chr3', 'hg38_chr1']
part2 = ['hg38_chr22', 'hg38_chr20', 'hg38_chr18', 'hg38_chr16', 'hg38_chr14', 'hg38_chr12', 'hg38_chr10', 'hg38_chr8', 'hg38_chr6', 'hg38_chr4',  'hg38_chr2']
chromoPartition = [part1, part2]

#LSTM1
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"

#LSTM11: as LSTM1 but at earlier training stage:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"


#LSTM4S; trained on all odd numbered hg38-chromos; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"


#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"


#LSTM4P needs a little special care:
chromosomeOrderList = ['hg38_part2_chr22', 'hg38_part2_chr21', 'hg38_part2_chr20', 'hg38_part2_chr19', 'hg38_part2_chr18', 'hg38_part2_chr17', 'hg38_part2_chr16', 'hg38_part2_chr15', 'hg38_part2_chr14', 'hg38_part2_chr13', 'hg38_part2_chr12', 'hg38_part2_chr11', 'hg38_part2_chr10', 'hg38_part2_chr9', 'hg38_part2_chr8', 'hg38_part2_chr7', 'hg38_part2_chr6', 'hg38_part2_chr5', 'hg38_part2_chr4', 'hg38_part2_chr3', 'hg38_part2_chr2', 'hg38_part2_chr1']
chromosomeDict = {'hg38_part2_chr22':[0,1e9], 'hg38_part2_chr21':[0,1e9], 'hg38_part2_chr20':[0,1e9], 'hg38_part2_chr19':[0,1e9], 'hg38_part2_chr18':[0,1e9], 'hg38_part2_chr17':[0,1e9], 'hg38_part2_chr16':[0,1e9], 'hg38_part2_chr15':[0,1e9], 'hg38_part2_chr14':[0,1e9], 'hg38_part2_chr13':[0,1e9], 'hg38_part2_chr12':[0,1e9], 'hg38_part2_chr11':[0,1e9], 'hg38_part2_chr10':[0,1e9], 'hg38_part2_chr9':[0,1e9], 'hg38_part2_chr8':[0,1e9], 'hg38_part2_chr7':[0,1e9], 'hg38_part2_chr6':[0,1e9], 'hg38_part2_chr5':[0,1e9], 'hg38_part2_chr4':[0,1e9], 'hg38_part2_chr3':[0,1e9], 'hg38_part2_chr2':[0,1e9], 'hg38_part2_chr1':[0,1e9]}
#Odd/even partition:
part1 = ['hg38_part2_chr21', 'hg38_part2_chr19', 'hg38_cpart2_hr17', 'hg38_part2_chr15', 'hg38_part2_chr13', 'hg38_part2_chr11', 'hg38_part2_chr9', 'hg38_part2_chr7', 'hg38_part2_chr5', 'hg38_part2_chr3', 'hg38_part2_chr1']
part2 = ['hg38_part2_chr22', 'hg38_part2_chr20', 'hg38_part2_chr18', 'hg38_part2_chr16', 'hg38_part2_chr14', 'hg38_part2_chr12', 'hg38_part2_chr10', 'hg38_part2_chr8', 'hg38_part2_chr6', 'hg38_part2_chr4',  'hg38_part2_chr2']
chromoPartition = [part1, part2]



returnDict = stats.calculateAggrAccOnChromosPartition(rootOutput = rootOutput, chromosomeDict = chromosomeDict, chromoPartition = chromoPartition, dictionaryName = 'accuracyChromoByAnnoDictionary')



#make a bar plot of these results:
rootPredictModelList = [r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/notAvgRevCompl/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"]
modelNameList = ['LSTM1'
, 'LSTM4S'
, 'LSTM4'
, 'LSTM4P'
, 'LSTM11']
partitionNameList = ['odd', 'even']
rootOutputPlot = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/"
offsetValue = 0.5
legend_b = 1
returnDict = stats.makePlotOfAggrAccOnChromosPartitionSeveralModels(rootPredictModelList = rootPredictModelList, modelNameList = modelNameList, partitionNameList = partitionNameList, rootOutput = rootOutputPlot,  saveAtDpi = 300, offsetValue = offsetValue, legend_b = legend_b)

#make a tex able holding the results:
rowNames = modelNameList
colNames = ['part1', 'part2']
nameChangeDict = {'part1':'odd', 'part2':'even'}
captionText = 'Accuracy aggregatd on odd/even numbered chromosomes'
fileName = 'table_chromo_partition.txt'
stats.makeTexTable(inputDict = returnDict , rowColHeading = 'model/partition', rowNames = rowNames, columnNames = colNames, nameChangeDict = nameChangeDict, captionText = captionText, rootOutput = rootOutputPlot, fileName = fileName ) 
_________________________________________________________    

Step 3. Make tex-tables/plots (from the acc dict's created above)
_________________________________________________________    

#LSTM1
rootResults = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/'
captionText = 'LSTM1/notAvgRevCompl'
modelName = 'LSTM1'

#LSTM11
rootResults = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/'
captionText = 'LSTM11/notAvgRevCompl'
modelName = 'LSTM11'

#LSTM4S; trained on all odd numbered hg38-chromos; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM4S'
modelName = 'LSTM4S'

#LSTM4 (5)
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM4 (5)'

#LSTM4P
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM50P'


#GC/AT bias:
rootOutputBias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38"  
rootResults  = rootOutputBias + r"/GCbias/notAvgRevCompl/"
captionText = 'GC/AT content'

#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'mouseLSTM4'

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'mouseLSTM4 on mm10 (GRCm38)'

#Zebrafish
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'zebrafishLSTM4 on GRCz11'

#droso, w. train test split
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM4 on dm6 (r6.18)'

#droso, less trained (LSTM41), w. train test split
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM41 on dm6 (r6.18)'


#Yeast, w. train test split
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM4 on R64'

#Yeast, less trained (LSTM41), w. train test split
rootResults = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM41/trainTestSplit_80_20/notAvgRevCompl/"
captionText = 'LSTM41 on R64'


import cPickle as pickle

#Results per chromo/anno
loadFile = rootResults + 'accuracyByChromoAnnoDictionary' 
resultsDictChromo = pickle.load(open( loadFile, "rb"))

#Aggr over all chromos
loadFile = rootResults + 'accuracyByAnnoDictionary' 
resultsDict = pickle.load(open( loadFile, "rb"))

rootOutput = rootResults
fileName = 'table_chromo_anno.txt'

#for human (GC/AT too)
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
rowNames = chromosomeOrderList[::-1] #the chromos will appear in this order 
colNames = ['all', 'repeat', 'simple_repeat', 'repeatsGenomeSeq', 'cds', 'gene', 'introns', '3UTR', '5UTR']   #the models will appear in columns in this order
#part2/LSTM4P:
chromosomeOrderList = ['hg38_part2_chr22', 'hg38_part2_chr21', 'hg38_part2_chr20', 'hg38_part2_chr19', 'hg38_part2_chr18', 'hg38_part2_chr17', 'hg38_part2_chr16', 'hg38_part2_chr15', 'hg38_part2_chr14', 'hg38_part2_chr13', 'hg38_part2_chr12', 'hg38_part2_chr11', 'hg38_part2_chr10', 'hg38_part2_chr9', 'hg38_part2_chr8', 'hg38_part2_chr7', 'hg38_part2_chr6', 'hg38_part2_chr5', 'hg38_part2_chr4', 'hg38_part2_chr3', 'hg38_part2_chr2', 'hg38_part2_chr1']
colNames = ['all']

#for mouse:
chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
rowNames = chromosomeOrderList #the chromos will appear in this order 
colNames = ['all', 'repeatsGenomeSeq']   #the models will appear in columns in this order

#for zebrafish
chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
rowNames = chromosomeOrderList
colNames = ['all', 'repeatsGenomeSeq']   #the models will appear in columns in this order

#for droso
chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
rowNames = chromosomeOrderList
colNames = ['all']

#for yeast:
chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
rowNames = chromosomeOrderList
colNames = ['all', 'simpleRepeats']   #the models will appear in columns in this order


#wo final aggregation row:
stats.makeTexTable(inputDict = resultsDictChromo , rowColHeading = 'chr/annotation', rowNames = rowNames,
                                  columnNames = colNames, captionText = captionText, rootOutput = rootOutput, fileName = fileName )

#with final aggregation row:
stats.makeTexTable(inputDict = resultsDictChromo , rowColHeading = 'chr/annotation', rowNames = rowNames,
                                  columnNames = colNames, inputDict2 = resultsDict, captionText = captionText, rootOutput = rootOutput, fileName = fileName )


#Scatter-plot having chromo on x-axis and values by anno on y-axis:
#For human, hg38
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeOrderList = chromosomeOrderList[::-1]
modelName = modelName
nameChangeDict = {}
saveAtDpi = 300
stats.scatterplotAnnoByChromo(inputDict = resultsDictChromo, chromosomeOrderList = chromosomeOrderList, modelName = modelName, nameChangeDict = nameChangeDict, rootOutput = rootResults, saveAtDpi = saveAtDpi)

- - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - 
To gather acc's from several models and generate a corr tex-table
- - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - 

#Generate the input lists of the models data:
rootPredictModelList = []
modelFileList = []
modelNameList = []

#LSTM1
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileName ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
modelName = 'LSTM1'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)


#LSTM4S; trained on all odd numbered hg38-chromos; shares the convo part (word encoding) with LSTM1, else as LSTM4:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr184'
modelName = 'LSTM4S'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)


#LSTM4/5
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
modelName = 'LSTM4'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)


#LSTM4P
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199'
modelName = 'LSTM4P'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)


#LSTM11
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileName ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr15"
modelName = 'LSTM11'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)

#mouseLSTM4 on hg38
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
modelName = 'mouseLSTM4'

rootPredictModelList.append(rootOutput)
modelFileList.append(modelFileName)
modelNameList.append(modelName)



#place the resulting dict here:
rootOutputTheModels = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/TheModels/3Models/"

 
#For acc-by-chromo per model:
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeOrderList = chromosomeOrderList[::-1]
saveAtDpi = 300
addAvg_b = 1
avgLevel = 0.5212
resultsDictByChromoModel = stats.collectAccuracyChromosomesSeveralModels(rootOutput =rootOutputTheModels, 
                                  rootPredictModelList = rootPredictModelList, 
                                  modelFileList = modelFileList,
                                  modelNameList = modelNameList,
                                  chromosomeOrderList = chromosomeOrderList,
                                  plot_b = 1,
                                  addAvg_b = addAvg_b,
                                  avgLevel = avgLevel,
                                  saveAtDpi = saveAtDpi)

#make tex-table:
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
rowNames = chromosomeOrderList[::-1] #the chromos will appear in this order 
colNames = modelNameList #the models will appear in columns in this order
rootOutput = rootOutputTheModels
fileName = 'table_chromo_model.txt'
captionText = 'Acc by chromo, several models'

stats.makeTexTable(inputDict = resultsDictByChromoModel , rowColHeading = 'chr/model', rowNames = rowNames,
                                  columnNames = colNames, inputDict2 = {}, captionText = captionText, rootOutput = rootOutput, fileName = fileName )


#Similarly: For acc-by-anno per model:
annotationOrderList = ['all', 'repeat', 'simple_repeat', '3UTR', '5UTR', 'introns', 'cds', 'gene', 'repeatsGenomeSeq']
saveAtDpi = 300
resultsDictByAnnoModel = stats.collectAccuracyAnnotationsSeveralModels(rootOutput =rootOutputTheModels, 
                                  rootPredictModelList = rootPredictModelList, 
                                  modelFileList = modelFileList,
                                  modelNameList = modelNameList,
                                  annotationOrderList = annotationOrderList,
                                  plot_b = 1,
                                  saveAtDpi = saveAtDpi)

#make tex-table:
annotationOrderList = ['all', 'repeat', 'simple_repeat', '3UTR', '5UTR', 'introns', 'cds', 'gene', 'repeatsGenomeSeq']
rowNames = annotationOrderList #the chromos will appear in this order 
colNames = modelNameList #the models will appear in columns in this order
rootOutput = rootOutputTheModels
fileName = 'table_anno_model.txt'
captionText = 'Acc by anno, several models'

stats.makeTexTable(inputDict = resultsDictByAnnoModel , rowColHeading = 'anno/model', rowNames = rowNames,
                                  columnNames = colNames, inputDict2 = {}, captionText = captionText, rootOutput = rootOutput, fileName = fileName )


######################################################################################3

# Part 7. Fourier runs

######################################################################################3

_________________________________________________________

Base version: Fourier on segments
_________________________________________________________


rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/LSTM1/"

chrName = 'hg38_chr22'

#LSTM1
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/" + chrName + "/"


#LSTM4 on h19
rootModel = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/scratch/human/on_hg19/LSTM4/"
rootOutput = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/scratch/human/on_hg19/LSTM4/notAvgRevCompl/" + chrName + "/"
modelFileNameNN = r"modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150"

segmentLength = 1000000
genomeIdName = chrName + '_seg' + str(int(segmentLength))

augmentWithRevComplementary_b = 0 #!!!!!

#window lgth and stepsize used in generaing the avg prediction
windowLength = 5
stepSize = 1

#Param's for Fourier plots:
fourierWindowLength = 1000
fourierStop = 15000
fourierStep = 100
fourierRawPlotFrq = 10
                 

stats.computeFourierOnSegments(rootOutput = rootOutput,
                             modelFileName = modelFileNameNN,  
                             segmentLength = segmentLength,
                             genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                             averageRevComplementary_b = augmentWithRevComplementary_b,
                             windowLength = windowLength,
                             stepSize = stepSize,
                             fourierWindowLength = fourierWindowLength,
                             fourierStop = fourierStop,
                             fourierStep = fourierStep, 
                             fourierRawPlotFrq = fourierRawPlotFrq )
                             
                      
_________________________________________________________

Full version: Big run of Fourier in segments: across list of chromo's
_________________________________________________________
    

#Human

#LSTM1
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

#LSTM11: as LSTM1 but at earlier training stage:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr15"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!


#LSTM2: as LSTM1 but not trained w aug rev compl:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/LSTM2/notAvgRevCompl/"
modelFileNameNN ="modelLSTM__2Conv2LayerLstm_flanks200_win3_filters64And256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr196"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!


#LSTM4 on hg19
modelFileNameNN = r"modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150"
rootOutput = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/scratch/human/on_hg19/LSTM4/notAvgRevCompl/"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!


#LSTM4 on hg18 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20 (and a prediction was run on hg18 chr12)
#GC bias only uses the q-arrays, and we use the flanks 50 version then and not the 200: 
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

chromosomeOrderList = ['hg18_chr12']
#turns out that at ratioQcutoff = 0.9 it'll skip the centromeric stuff, so we lower it:
ratioQcutoff = 0.7



#AT bias
#this is to get the qual arrays from the model-pred run:
modelFileNameNN = r"modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'ATbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/ATbias/notAvgRevCompl/"
forATorGCbias_b = 1 #!!!!!!!!!!!!

#GC bias
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
#modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!


chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']

#chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']

chromosomeOrderList = ['hg38_chr22']

chromosomeOrderList = ['hg38_chr20']



#Droso

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
#chromosomeOrderList = ['r6.18_chr2L', 'r6.18_chr2R']


#GC bias
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!




#Mouse

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']

chromosomeOrderList = [ 'm38_chr11']

#GC bias
#this is to get the qual arrays from the model-pred run:
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']


#Mouse mm9 (same settings as the human LSTM4):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr200'

#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

chromosomeOrderList = ['mm9_chr11', 'mm9_chr7','mm9_chr5', 'mm9_chr3']

#GC bias
#this is to get the qual arrays from the model-pred run:
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr200'
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/mouseLSTM5/trainTestSplit_80_20/notAvgRevCompl/"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!




#Zebrafish

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 0 #!!!!!!!!!!!!

chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']

chromosomeOrderList = ['GRCz11_chr1']


#GC bias
chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

#AT bias
chromosomeOrderList = [ 'GRCz11_chr3', 'GRCz11_chr5',  'GRCz11_chr7', 'GRCz11_chr11']
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'ATbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/ATbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!


#Human on alternative genome sequences: chromo's downloaded directly from UCSC:

#GC bias
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38_ucscDirect/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['hg38_chr19', 'hg38_chr17', 'hg38_chr5']


#on hg18:
#this is to get the qual arrays from the model-pred run (model trained on h38 but applied to hg18 chr12):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/GCbias/notAvgRevCompl/"  
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['hg18_chr12']

#turns out that at ratioQcutoff = 0.9 it'll skip the centromeric stuff, so we lower it:
ratioQcutoff = 0.7

dumpFourier_b = 1
dumpFileNamePrefix = ''



#abuse the ATorGCbias-set up for MNase data, human:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#this is to get the "predReturns" for the mnase peaks/amplitudes:
modelFileName_forATorGCbias = 'MNase'
rootOutput_forATorGCbias = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/MNase/"  
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['hg18_chr12']

#turns out that at ratioQcutoff = 0.9 it'll skip the centromeric stuff, so we lower it:
ratioQcutoff = 0.7



#abuse the ATorGCbias-set up for SINE Alu annotation, human:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#this is to get the "predReturns" for the mnase peaks/amplitudes:
modelFileName_forATorGCbias = 'SINEAlu'
rootOutput_forATorGCbias = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg18/Other/"  
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['hg18_chr12']

#turns out that at ratioQcutoff = 0.9 it'll skip the centromeric stuff, so we lower it:
ratioQcutoff = 0.7




#abuse the ATorGCbias-set up for MNase data, mouse:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'MNase'
rootOutput_forATorGCbias = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/MNase/"  
rootInput_forATorGCbias = rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['m38_chr1']

#turns out that at ratioQcutoff = 0.9 it'll skip the centromeric stuff, so we lower it:
ratioQcutoff = 0.7

#And another abuse the ATorGCbias-set up for MNase data, mouse, here on mm9:
(lost, apparently)



#Yeast

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"

#rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_ proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
#modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr8"

rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':[0,1e8], 'R64_chr2':[0,1e8], 'R64_chr3':[0,1e8], 'R64_chr4':[0,1e8], 'R64_chr5':[0,1e8], 'R64_chr6':[0,1e8], 'R64_chr7':[0,1e8], 'R64_chr8':[0,1e8],'R64_chr9':[0,1e8], 'R64_chr10':[0,1e8], 'R64_chr11':[0,1e8], 'R64_chr12':[0,1e8],'R64_chr13':[0,1e8], 'R64_chr14':[0,1e8], 'R64_chr15':[0,1e8], 'R64_chr16':[0,1e8]}

#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

#GC bias
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/GCbias/notAvgRevCompl/"
rootInput_forATorGCbias =rootOutput_forATorGCbias
forATorGCbias_b = 1 #!!!!!!!!!!!!



#General settings:

#segmentLength = 1000000
segmentLength = 100000

augmentWithRevComplementary_b = 0  #!!!!!

#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
                      
#Param's for Fourier plots:
fourierWindowLength = 1000
fourierStart = 200
fourierStop = 30000 #!!!!!!!!!!!!
fourierStep = 100
fourierRawPlotFrq = 10
shuffle_b = 0 #!!!!!!!!!!!!!!!!!!!!!
randomizeDisqualified_b = 0 #!!!!!!!!!!!!!
randomizingByShuffle_b = 0 #!!!!!!!!!

fullGC_b = 0 #!!!!!!!!!!!!!
#rootInput_forATorGCbias = '' #r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/GCbias/notAvgRevCompl/"
#rootOutput_forATorGCbias = '' #r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/GCbias_full/notAvgRevCompl/"


#Which input to use:
inputArrayType = 0 # 1: ref base prob's; 0: pred returns
#plotOnlyNorm_b = 1 #default:1 

#fourierStartList = [200, 200, 40000]
#fourierStopList = [20000, 45000, 140000]
#fourierWindowLengthList = [1000, 1000, 5000]


#fourierStartList = [ 200, 40000]
#fourierStopList = [ 45000, 140000]
#fourierWindowLengthList = [ 1000, 5000]

#Yeast
fourierStep = 10
fourierRawPlotFrq = 5
fourierStartList = [20, 20, 4000]
fourierStopList = [2000, 4500, 14000]
fourierWindowLengthList = [100, 100, 500]

#hg18 and others ...
#fourierStartList = [200]
#fourierStopList = [20000]
#fourierWindowLengthList = [1000]

ratioQcutoff = 0.9 #0.7 w hg18
dumpFourier_b = 0
dumpFileNamePrefix = ''
for i in range(len(fourierStartList)):
    fourierStart = fourierStartList[i]
    fourierStop = fourierStopList[i]
    fourierWindowLength = fourierWindowLengthList[i]
    stats.computeFourierChromosomes(chromosomeOrderList = chromosomeOrderList,
                                    rootOutput = rootOutput,
                                    modelFileName = modelFileNameNN,  
                                    segmentLength = segmentLength,
                                    inputArrayType = inputArrayType,
                                    averageRevComplementary_b = augmentWithRevComplementary_b,
                                    ratioQcutoff = ratioQcutoff,
                                    windowLength = windowLength,
                                    stepSize = stepSize,
#                                    plotOnlyNorm_b = plotOnlyNorm_b,
                                    fourierWindowLength = fourierWindowLength,
                                    fourierStart = fourierStart,
                                    fourierStop = fourierStop,
                                    fourierStep = fourierStep, 
                                    fourierRawPlotFrq = fourierRawPlotFrq,
                                    shuffle_b = shuffle_b,
                                    randomizeDisqualified_b =randomizeDisqualified_b,
                                    randomizingByShuffle_b = randomizingByShuffle_b,
                                    forATorGCbias_b = forATorGCbias_b, 
                                    rootOutput_forATorGCbias= rootOutput_forATorGCbias,
                                    rootInput_forATorGCbias = rootInput_forATorGCbias,
                                    fullGC_b = fullGC_b,
                                    dumpFourier_b = dumpFourier_b,
                                    dumpFileNamePrefix = dumpFileNamePrefix, 
                                    modelFileName_forATorGCbias = modelFileName_forATorGCbias)




__________________________________________________________________________________________________________________

Additional: Fourier on external model's prediction across list of chromo's
__________________________________________________________________________________________________________________
    

#Human      


#Markov model:
rootOutput  = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/Markovs/Markov14/avgRevCompl/"
modelFileName = 'Bidirectional_Markov_win14'              
augmentWithRevComplementary_b = 1 #!!!!!
           

#k-mer model:
rootOutput = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/'

#place holders
modelPredicitionFileNameDict = {}
augmentWithRevComplementary_b = 0

kMer_b = 1
kModel = 5
modelFileName = '5-mer'



#chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']

chromosomeOrderList = ['hg38_chr22']

segmentLength = 1000000


#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
                      
#Param's for Fourier plots:
fourierWindowLength = 1000
fourierStart = 200
fourierStop = 20000
fourierStep = 100
fourierRawPlotFrq = 10

inputArrayType = 1 #1: ref base prob's; 0: pred returns
                      
shuffle_b = 0
forATorGCbias_b = 0
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
modelFileName_forATorGCbias = ''
                           
stats.computeFourierChromosomes(chromosomeOrderList = chromosomeOrderList,  
                            rootOutput = rootOutput,
                             modelFileName = modelFileName,  
                             segmentLength = segmentLength,
                              inputArrayType = inputArrayType,
                             averageRevComplementary_b = augmentWithRevComplementary_b,
                             windowLength = windowLength,
                             stepSize = stepSize,
                             fourierWindowLength = fourierWindowLength,
                             fourierStart = fourierStart,
                             fourierStop = fourierStop,
                             fourierStep = fourierStep, 
                             fourierRawPlotFrq = fourierRawPlotFrq,
                             shuffle_b = shuffle_b,
                             forATorGCbias_b = forATorGCbias_b, 
                             rootInput_forATorGCbias = rootInput_forATorGCbias,
                             rootOutput_forATorGCbias= rootOutput_forATorGCbias,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias)



*******************************************************************************************
** Fouriers on randomly picked segments (for reassuring that not computational artefact)
*******************************************************************************************


#Human
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'

#To obtain a complete prediction array across all segments 
genomeIdName = 'hg38_chr19'
rootOutput = rootOutput + r'/' + genomeIdName + r'/' 
nrSegments = 500 #just higher than max segment nr 
segmentLength = 1000000                 
#for GCbias case or not: 
for_GCbias_b = 1
inputArrayType = 0
rootGCbias = '/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/GCbias/notAvgRevCompl/'
rootGCbias = rootGCbias + r'/' + genomeIdName + r'/' 
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileName, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, for_GCbias_b = for_GCbias_b, rootGCbias = rootGCbias)                     

rootOutput_alt = rootOutput + 'FourierRandomSegments/'

fourierParamList = [[100, 2000, 10, 100, 100000], [200, 25000, 100, 1000, 1000000]]


#Mouse
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

#To obtain a complete prediction array across all segments 
genomeIdName = 'm38_chr1'
rootOutput = rootOutput + r'/' + genomeIdName + r'/' 
nrSegments = 500 #just higher than max segment nr 
segmentLength = 1000000                 
#for GCbias case or not: 
for_GCbias_b = 0
inputArrayType = 1
rootGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/GCbias/notAvgRevCompl/"
rootGCbias = rootGCbias + r'/' + genomeIdName + r'/' 
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileName, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, for_GCbias_b = for_GCbias_b, rootGCbias = rootGCbias)                     

rootOutput_alt = rootOutput + 'FourierRandomSegments/'

fourierParamList = [[100, 2000, 10, 100, 100000], [100, 10000, 10, 500, 500000], [200, 25000, 100, 1000, 1000000]]


#Droso
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr168"

#To obtain a complete prediction array across all segments 
genomeIdName = 'r6.18_chr3R'
rootOutput = rootOutput + r'/' + genomeIdName + r'/' 
nrSegments = 500 #just higher than max segment nr 
segmentLength = 1000000                 
#for GCbias case or not: 
for_GCbias_b = 0
inputArrayType = 1
rootGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/GCbias/notAvgRevCompl/"
rootGCbias = rootGCbias + r'/' + genomeIdName + r'/' 
modelFileName_forATorGCbias = 'GCbias'
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileName, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, for_GCbias_b = for_GCbias_b, rootGCbias = rootGCbias)                     

rootOutput_alt = rootOutput + 'FourierRandomSegments/'

fourierParamList = [[100, 2000, 10, 100, 100000], [100, 10000, 10, 500, 500000], [200, 25000, 100, 1000, 1000000]]


#Zebrafish
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName ="modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr199"

#To obtain a complete prediction array across all segments 
genomeIdName = 'GRCz11_chr6'
rootOutput = rootOutput + r'/' + genomeIdName + r'/' 
nrSegments = 500 #just higher than max segment nr 
segmentLength = 1000000                 
#for GCbias case or not: 
for_GCbias_b = 1
inputArrayType = 0
rootGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/GCbias/notAvgRevCompl/"
rootGCbias = rootGCbias + r'/' + genomeIdName + r'/' 
modelFileName_forATorGCbias = 'GCbias'
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileName, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, for_GCbias_b = for_GCbias_b, rootGCbias = rootGCbias)                     

rootOutput_alt = rootOutput + 'FourierRandomSegments/'

fourierParamList = [[100, 5000, 10, 100, 100000], [100, 25000, 10, 500, 500000], [200, 50000, 100, 1000, 1000000]]




#Yeast
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileName = "modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr59"

#To obtain a complete prediction array across all segments 
genomeIdName = 'R64_chr4'
rootOutput = rootOutput + r'/' + genomeIdName + r'/' 
nrSegments = 20 #just higher than max segment nr 
segmentLength = 100000                 
#for GCbias case or not: 
for_GCbias_b = 1
inputArrayType = 0
rootGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/GCbias/notAvgRevCompl/"
rootGCbias = rootGCbias + r'/' + genomeIdName + r'/' 
#call assembly
predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = stats.assemblePredictArrayFromSegments(rootOutput = rootOutput, modelFileName = modelFileName, genomeIdName = genomeIdName, nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength, for_GCbias_b = for_GCbias_b, rootGCbias = rootGCbias)                     

rootOutput_alt = rootOutput + 'FourierRandomSegments/'

fourierParamList = [[100, 2000, 10, 100, 100000], [200, 15000, 50, 500, 500000], [200, 25000, 100, 1000, 1000000]]



#Call Fourier transf:
inputArray = predArray

modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = rootGCbias + 'FourierRandomSegments/'

forATorGCbias_b = 1
stats.computeFourierRandomSegments(rootOutput = rootOutput_alt,
                         modelFileName = modelFileName,                        
                         inputArray = inputArray,
                         labelArray = labelArray,
                         qualArray = qualArray,
                         inputArrayType = inputArrayType,
                         genomeIdName = genomeIdName, #can be anything, just an id of the inputArray
                         windowLength = 1, #if inputArray is arrived at by a sliding-window approach 
                         stepSize = 1, #stride if inputArray is arrived at by a sliding-window approach 
                         nrSamples = 10,
                         averageRevComplementary_b = 0,
                         fourierParamList = fourierParamList,
                         fourierRawPlotFrq = 10,
                         shuffle_b = 0,
                         forATorGCbias_b = forATorGCbias_b,
                         rootOutput_forATorGCbias = rootOutput_forATorGCbias,
                         modelFileName_forATorGCbias = modelFileName_forATorGCbias,
                         dumpFourier_b = 0, 
                         ratioQcutoff = 0.9)
                         

*******************************************************************************************
** Computing Fouriers on some test cases -- simply to see what the output looks like
*******************************************************************************************

#We use the set-up on some test-cases, ie contructed arrays (eg a single 1Mb segment) to 
#get some concrete look into the shape of the Fourier outputs. Since we use the set-up, file names
#and structure must follow that from eg model output.



#Case 1. 
#Construct a 1Mb array having a 300 bp long sequence of 1 for every 3000 bp's, and equally spaced:
L = 1000000
testArray1 = np.zeros(shape = L)
for i  in range(L):
    if i%3000 == 0 and i < L - 300:
        testArray1[i:(i+300)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_300_pr_3000'
genomeId = 'NN1'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray1, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))


#Case 2. 
#As Case 1, but we place the 300 bp seg's randomly within the 3000 seg's 
L = 1000000
testArray2 = np.zeros(shape = L)
for i  in range(L):
    if i%3000 == 0 and i < L - 300:
        rndStart = np.random.randint(i, i+2700, 1)
        rndStart = int(rndStart)
        testArray2[rndStart:(rndStart+300)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_300_randomly_pr_3000'
genomeId = 'NN2'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray2, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))


#Case 3.
#As case 1, but with 10000 bps instead of 3000 bps: 
#Construct a 1Mb array having a 300 bp long sequence of 1 for every 10000 bp's, and equally spaced:
L = 1000000
testArray3 = np.zeros(shape = L)
for i  in range(L):
    if i%10000 == 0 and i < L - 300:
        testArray3[i:(i+300)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_300_pr_10000'
genomeId = 'NN3'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray3, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))


#Case 4. 
#As Case 3, but we place the 300 bp seg's randomly within the 3000 seg's 
L = 1000000
testArray4 = np.zeros(shape = L)
for i  in range(L):
    if i%10000 == 0 and i < L - 300:
        rndStart = np.random.randint(i, i+9700, 1)
        rndStart = int(rndStart)
        testArray4[rndStart:(rndStart+300)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_300_randomly_pr_10000'
genomeId = 'NN4'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray4, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))


#Case 5. 
#As case 1 but with 200 bps instead of 300 bps. 
#Construct a 1Mb array having a 200 bp long sequence of 1 for every 3000 bp's, and equally spaced:
L = 1000000
testArray5 = np.zeros(shape = L)
for i  in range(L):
    if i%3000 == 0 and i < L - 200:
        testArray5[i:(i+200)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_200_pr_3000'
genomeId = 'NN5'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray5, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))



#Case 6. 
#As case 1 but with 400 bps instead of 300 bps. 
#Construct a 1Mb array having a 400 bp long sequence of 1 for every 3000 bp's, and equally spaced:
L = 1000000
testArray6 = np.zeros(shape = L)
for i  in range(L):
    if i%3000 == 0 and i < L - 400:
        testArray6[i:(i+400)] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_400_pr_3000'
genomeId = 'NN6'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray6, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))




#Case 10. 
#Construct a 1-per-10 periodic 1Mb array:
L = 1000000
testArray10 = np.zeros(shape = L)
for i  in range(L):
    if i%10 == 0 and i < L - 10:
        testArray10[i] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_1_pr_10'
genomeId = 'NN10'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray10, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))

#Case 11. 
#As case 10 , but with the 1 placed randomly within the 10bps: 
L = 1000000
testArray11 = np.zeros(shape = L)
for i  in range(L):
    if i%10 == 0 and i < L - 10:
        rndStart = np.random.randint(i, i+10, 1)
        rndStart = int(rndStart)
        testArray11[rndStart] = 1

#Dump this array:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/noOrg/FourierExperiment/"
modelFileName = 'bool_1_randomly_pr_10'
genomeId = 'NN11'
genomeIdNameSeg = genomeId + '_seg' + str(int(L)) + '_segment' + str(0)
averageRevComplementary_b = 0
windowLength = 1
stepSize = 1
dumpFile = rootOutput + genomeId + r'/' +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  =  testArray11, 0,0,0,0,0
cPickle.dump((avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg), open(dumpFile, "wb"))

#Dump a qualifiedArray too (just 1's):
qualArray = np.ones(shape  = L)
dumpFile = rootOutput + genomeId + r'/' + modelFileName + '_qualifiedArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
cPickle.dump(qualArray, open(dumpFile, "wb"))



#Run the Fourier on these two arrays:
chromosomeOrderList = ['NN6']

segmentLength = L


#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
                      
#Param's for Fourier plots:
fourierWindowLength = 1000
fourierStart = 45000
fourierStop = 120000
fourierStep = 100
fourierRawPlotFrq = 10

inputArrayType = 0 #1: ref base prob's; 0: pred returns
                      
shuffle_b = 0
forATorGCbias_b = 0
rootOutput_forATorGCbias = ''
rootInput_forATorGCbias = ''
modelFileName_forATorGCbias = ''
                           
stats.computeFourierChromosomes(chromosomeOrderList = chromosomeOrderList,  
                            rootOutput = rootOutput,
                             modelFileName = modelFileName,  
                             segmentLength = segmentLength,
                              inputArrayType = inputArrayType,
                             averageRevComplementary_b = augmentWithRevComplementary_b,
                             windowLength = windowLength,
                             stepSize = stepSize,
                             fourierWindowLength = fourierWindowLength,
                             fourierStart = fourierStart,
                             fourierStop = fourierStop,
                             fourierStep = fourierStep, 
                             fourierRawPlotFrq = fourierRawPlotFrq,
                             shuffle_b = shuffle_b,
                             forATorGCbias_b = forATorGCbias_b, 
                             rootInput_forATorGCbias = rootInput_forATorGCbias,
                             rootOutput_forATorGCbias= rootOutput_forATorGCbias,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias)
                         
                       

########################################################################################################
# Part 8: One-off's: 
########################################################################################################

_________________________________________________________
    
Part 8.0: to get genome seq for assembly split over chromo's; read in probs from external source
_________________________________________________________
    

import dnaNet_dataGen as dataGen
chromoNameBound = 1000
rootGenome = '/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'
genomeFileName = 'GCF_000001405.38_GRCh38.p12_genomic_filter.fna'
genomeShortName = 'hg38'
rootOut = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
stats.dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = genomeFileName, genomeShortName = genomeShortName, rootOut = rootOut, chromoNameBound = chromoNameBound)


#Read in prob's from external:
fileName = r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'
positionArrayMarkov, predArrayMarkov = stats.readInProbsFromExternal(fileName)


_________________________________________________________
    
Part 8.1: SNPs
_________________________________________________________
  
#############################
#For NN model 
#############################
  
#Assumes that we have obtained a predArray (from some model). To obtain it from a segmented prediction across
#eg a chromo (for NN only); we have to provide also the encoded genomic data that were used for the prediction
#let us do that first:
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9]}


rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
chromoName = 'hg38_chr22'
fileName = chromoName + ".txt"
fileGenome = rootGenome +fileName


#Now fetch the assembled pred-array for the model:  
rootOutNN = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
rootOutNN = rootOutNN + chromoName + r'/'
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
augmentWithRevComplementary_b = 0
segmentLength = 1000000
nrSegments = 100
predArrayNN, labelArrayNN, qualArrayNN, sampledPositionsNN, sampledPositionsBooleanNN = stats.assemblePredictArrayFromSegments(rootOutput = rootOutNN, modelFileName = modelFileNameNN, genomeIdName = chromoName,  nrSegments = nrSegments, augmentWithRevComplementary_b = augmentWithRevComplementary_b, segmentLength = segmentLength)                     


#Get the snp-data:
import snpAnalysis as snp

#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'
chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'

snpInfoArray = snp.readSNPdata(rootSnp + fileName)

#Then to get the SNP triangle plot on NN model (optionally the prob's at the SNPs written to file):
rootOut = rootOutNN + 'SNP/'
modelFileName = 'LSTM1'
genomeIdName = chromoName
snpIdName = '1000Gs'
snpIndexOffset = 1
predictionArray = predArrayNN
qualArray = qualArrayNN
labelArray = labelArrayNN
title = 'LSTM1 '
startAtPosition, endAtPosition = chromosomeDict[chromoName] #id to what was used for the encoding!
writeOut_b = 0 #if 1 will write out the model's prob's to file
probsAtSnp_1000G = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#get 2d-histo:
saveAtDpi = 300
snp.snpHisto2D(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdName = snpIdName, probsAtSnp = probsAtSnp_1000G, title = 'LSTM1', saveAtDpi = saveAtDpi)
#for later: randomized alt allele:
randomizeAlt_b = 1
probsAtSnp_1000G_rndAlt = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, randomizeAlt_b = randomizeAlt_b)


#Get density plot of diff of prob's of ref base and allele (log of, if wanted) for 1000G, clin-var and cosmics: 
#for the Cosmic and clinVar snp's -- and to get a density of all three in one plot, run the above code
#for the 1000Gs and then the same for the other sets:
fileName = 'clinvar_20200310_filter_INDEL.vcf'
snpInfoArray_Clin = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'ClinVar'
probsAtSnp_clinVar = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_Clin, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#for later: randomized alt allele:
randomizeAlt_b = 1
probsAtSnp_clinVar_rndAlt = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_Clin, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, randomizeAlt_b = randomizeAlt_b)


#and

fileName = 'CosmicNonCodingVariants.vcf'
snpInfoArray_CosmicNC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, non-coding'
probsAtSnp_cosmicNC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicNC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#for later: randomized alt allele:
randomizeAlt_b = 1
probsAtSnp_cosmicNC_rndAlt = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicNC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, randomizeAlt_b = randomizeAlt_b)


#and

fileName = 'CosmicCodingMuts.vcf'
snpInfoArray_CosmicC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, coding'
probsAtSnp_cosmicC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#for later: randomized alt allele:
randomizeAlt_b = 1
probsAtSnp_cosmicC_rndAlt= snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositionsNN,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title, randomizeAlt_b = randomizeAlt_b)



#and, maybe, a 'random set' even for various annotations (the call is similar to the call of assemblePredictArrayFromSegments above):
chromoName = 'hg38_chr22'
pctPositions = 0.01
rootOutput = rootOutNN
#modelFileName = modelFileNameNN
#segmentLength = 1000000
#genomeIdName = chromoName #for saving the prediction array; could eg be a chromo name
#nrSegments = 100 #encodedGenomeData,
#augmentWithRevComplementary_b = 0
windowLength = 1
stepSize = 1
annotationDict = {}
startAtPostion = 10500000 #flank size should not be 'corrected' for

assemblePredictArrayFromSegmentsOutput = predArrayNN, labelArrayNN, qualArrayNN, sampledPositionsNN, sampledPositionsBooleanNN
  
probsBackground = stats.makeRandomProbRefProbAltArray(pctPositions = pctPositions,
                                                      assemblePredictArrayFromSegmentsOutput = assemblePredictArrayFromSegmentsOutput,
#                                  rootOutput = rootOutput, 
#                        modelFileName = modelFileName,
#                        segmentLength = segmentLength,
#                        genomeIdName = genomeIdName , #for saving the prediction array; could eg be a chromo name
#                        nrSegments = nrSegments, #encodedGenomeData,
#                        augmentWithRevComplementary_b =  augmentWithRevComplementary_b,
#                        annotationDict = annotationDict,
                        startAtPosition = startAtPostion,
#                        repeatComplement_b = repeatComplement_b, 
#                        repeatAnnoTypes = repeatAnnoTypes,
                        windowLength = windowLength,
                        stepSize = stepSize)

probsBackground  = np.asarray(probsBackground)

#if SNP per annotation wanted:
rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
import cPickle as pickle
for annoType in ['cds','repeat']:
    annotationFile = rootAnnotationFiles + chromoName + '_annotationArray_' + annoType
    print annotationFile
    annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))

#For 1000G's and Cosmic NC we want the non-repeat parts:    
repeatComplement_b = 1 #1: get all the non-repeat positions for ..
repeatAnnoTypes = ['repeat']
probsAtSnp_1000G_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_1000G, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_1000G_noRepeats = probsAtSnp_1000G_noRepeats_dict['repeat']
probsAtSnp_cosmicNC_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_cosmicNC, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_cosmicNC_noRepeats = probsAtSnp_cosmicNC_noRepeats_dict['repeat']


#For the backgrund (ie random positions) we want both the cds and the non-repeats:
probsBackgroundDict = snp.probsAtSnpAnnos(probsAtSnp = probsBackground, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)

for k in probsBackgroundDict:
    print probsBackgroundDict[k].shape

probsBackgroundAll = probsBackgroundDict['all']
probsBackgroundNoRepeat = probsBackgroundDict['repeat']
probsBackgroundCds = probsBackgroundDict['cds']

#At long last ... the call to the density plot fct:
snpIdNameList = ['1000Gs','1000Gs, non-repeats', 'Cosmic, NC', 'Cosmic, NC, non-repeats', 'Cosmic, coding', 'ClinVar', 'Random, all', 'Random, cds' ]
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_1000G_noRepeats, probsAtSnp_cosmicNC, probsAtSnp_cosmicNC_noRepeats, probsAtSnp_cosmicC, probsAtSnp_clinVar , probsBackgroundAll, probsBackgroundCds]
colorCntList = [0,0, 1,1, 2,3,4,5]
linestyleList = ['-', '-.', '-', '-.', '-', '-', ':', ':' ]
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList)


snpIdNameList = ['1000Gs', 'Cosmic, NC',  'Cosmic, coding', 'ClinVar', 'Random, all', 'Random, cds' ]
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_cosmicNC,  probsAtSnp_cosmicC, probsAtSnp_clinVar , probsBackgroundAll, probsBackgroundCds]
colorCntList = [0, 1, 2,3,4,5]
linestyleList = ['-',  '-',  '-', '-', ':', ':' ]
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList)


snpIdNameList = ['1000Gs','1000Gs, non-repeats', 'Cosmic, NC', 'Cosmic, NC, non-repeats','Random, all', 'Random, cds' ]
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_1000G_noRepeats, probsAtSnp_cosmicNC, probsAtSnp_cosmicNC_noRepeats,  probsBackgroundAll, probsBackgroundCds]
colorCntList = [0,0, 1,1,4,5]
linestyleList = ['-', '-.', '-', '-.', ':', ':' ]
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList)


#plot including: 
snpIdNameList = ['1000Gs', 'Cosmic, non-coding',  'Cosmic, coding',  'ClinVar', 'Random, all', 'Random, cds']
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_cosmicNC, probsAtSnp_cosmicC, probsAtSnp_clinVar, probsBackgroundAll, probsBackgroundCds]
colorCntList = [0, 1,2, 3, 4, 5]
linestyleList = ['-',   '-', '-', '-',   '--', '--'  ]
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
returnSnpData_b = 1
snpDiffDict  = snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList, returnSnpData_b = returnSnpData_b)


#---------------------------
#Randomizing the alt allele 
#---------------------------

#Let us do this for the 1000Gs' case, the Cosmic C and ClinVar (Cosmic non+coding appears very similar to 1000Gs). First fetch the 
#snp-info with randomized alt: this was done above.
#Then  call the plotting:
snpIdNameList = ['1000Gs', '1000Gs, rnd. alt.', 'Cosmic, coding', 'Cosmic, coding, rnd. alt.', 'ClinVar', 'ClinVar, rnd. alt.']
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_1000G_rndAlt, probsAtSnp_cosmicC, probsAtSnp_cosmicC_rndAlt, probsAtSnp_clinVar, probsAtSnp_clinVar_rndAlt ]
colorCntList = [0, 0, 2,2, 3, 3]
linestyleList = ['-', ':',  '-', ':', '-', ':' , '-', ':']
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
returnSnpData_b = 1
makeDiffDensity_b = 1
snpDiffDict  = snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList, returnSnpData_b = returnSnpData_b, makeDiffDensity_b = makeDiffDensity_b)


#Let's plot the same, but restricting to non-repeat positions:
#Get annotation:
rootAnnotationFiles = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
import cPickle as pickle
for annoType in ['cds','repeat']:
    annotationFile = rootAnnotationFiles + chromoName + '_annotationArray_' + annoType
    print annotationFile
    annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))

#For 1000G's etc we want the non-repeat parts:    
repeatComplement_b = 1 #1: get all the non-repeat positions for ..
repeatAnnoTypes = ['repeat']
probsAtSnp_1000G_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_1000G, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_1000G_noRepeats = probsAtSnp_1000G_noRepeats_dict['repeat']
probsAtSnp_1000G_rndAlt_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_1000G_rndAlt, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_1000G_rndAlt_noRepeats = probsAtSnp_1000G_rndAlt_noRepeats_dict['repeat']

probsAtSnp_cosmicC_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_cosmicC, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_cosmicC_noRepeats = probsAtSnp_cosmicC_noRepeats_dict['repeat']
probsAtSnp_cosmicC_rndAlt_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_cosmicC_rndAlt, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_cosmicC_rndAlt_noRepeats = probsAtSnp_cosmicC_rndAlt_noRepeats_dict['repeat']

probsAtSnp_clinVar_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_clinVar, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_clinVar_noRepeats = probsAtSnp_clinVar_noRepeats_dict['repeat']
probsAtSnp_clinVar_rndAlt_noRepeats_dict = snp.probsAtSnpAnnos(probsAtSnp = probsAtSnp_clinVar_rndAlt, sampledPositions = sampledPositionsNN , startAtPosition = startAtPosition , annotationDict = annotationDict, repeatComplement_b = repeatComplement_b, repeatAnnoTypes = repeatAnnoTypes)
probsAtSnp_clinVar_rndAlt_noRepeats = probsAtSnp_clinVar_rndAlt_noRepeats_dict['repeat']


#Then  call the plotting:
snpIdNameList = ['1000Gs, no repeats', '1000Gs, rnd. alt., no repeats', 'Cosmic, coding, no repeats', 'Cosmic, coding, rnd. alt., no repeats', 'ClinVar, no repeats', 'ClinVar, rnd. alt., no repeats']
probsAtSnpList = [probsAtSnp_1000G_noRepeats, probsAtSnp_1000G_rndAlt_noRepeats, probsAtSnp_cosmicC_noRepeats, probsAtSnp_cosmicC_rndAlt_noRepeats, probsAtSnp_clinVar_noRepeats, probsAtSnp_clinVar_rndAlt_noRepeats ]
colorCntList = [0, 0, 2, 2, 3, 3]
linestyleList = ['-', ':',  '-', ':', '-', ':' , '-', ':']
log_b = 0
saveAtDpi = 300
#title = 'LSTM1 w backgound'
returnSnpData_b = 1
makeDiffDensity_b = 1
snpDiffDict  = snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = title, log_b = log_b, saveAtDpi = saveAtDpi, colorCntList = colorCntList,  linestyleList = linestyleList, returnSnpData_b = returnSnpData_b, makeDiffDensity_b = makeDiffDensity_b)


#############################
#For a k-mer model:
#############################

import snpAnalysis as snp
#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'
chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'

snpInfoArray = snp.readSNPdata(rootSnp + fileName)

#Need the genome seq/encoded:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
fileName = r"hg38_chr22.txt"
fileGenome = rootGenome +fileName

#Read in data from genome and get it encoded:
exonicInfoBinaryFileName = ''
chromoNameBound = 1000
startAtPosition = 10500000
endAtPosition = 3e9 #some big number
outputEncoded_b = 1
outputEncodedOneHot_b = 1
outputEncodedInt_b = 0
outputAsDict_b = 0
outputGenomeString_b = 1 #!!!
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


#There are two ways: 1) call the frq-model and have it predict on a set of positions (possibly all) 
#e.g. an entire chromosome, and write these results to a file and then fetch the corresponding pred-arrays, or
#2) simply avoid the write-to-file part (appears to be much easier): 

1): Here we have stored the predictions of a k-mer model for human chromo 22, and then fetch the corr pred-arrays:

rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/onOldCodeVersion/human/GRCh38.p12/chr22/'  
  
k=5
fileFrq = "frqModel_chr22_k" + str(k) + ".txt"
fileNameFrq = rootFrq + fileFrq
resDictFrq = stats.frqM.readResults(fileNameFrq)
#We get the pred-arrays here for some previously obtained positions (eg from a NN-model, as right above)
getAllPositions_b = 0
samplePositions = sampledPositionsNN 
sampledPositionsBoolean =  0 #not in use
sampleGenomeSequence =  encodedGenomeData[3]
predArrayFrq, qualArrayFrq, samplePositionsFrq, samplePositionsIndicatorArray, corrLabelArray  = stats.getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence, labelArray = labelArrayNN, fileGenome = '', exonicInfoBinaryFileName = '', chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel =resDictFrq, k = k)

2) Just get the pred-arrays directly:

rootFrq = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/"
k=5
fileFrq = "frqModel_k" + str(k) + ".txt"
fileNameFrq = rootFrq + fileFrq
resDictFrq = stats.frqM.readResults(fileNameFrq)
#As in case 1) we get the pred-arrays here for some previously obtained positions (eg from a NN-model, as right above)
getAllPositions_b =  0
samplePositions = sampledPositionsNN 
sampledPositionsBoolean = 0 #not in use
sampleGenomeSequence =  encodedGenomeData[3]  #from call to stats.dataGen.encodeGenome right above
predArrayFrq, qualArrayFrq, samplePositionsFrq, samplePositionsIndicatorArray, corrLabelArray  = stats.getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence, labelArray = labelArrayNN, fileGenome = '', exonicInfoBinaryFileName = '', chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel = resDictFrq, k = k)


#Then to get the SNP triangle plot on k-mer model (optionally the prob's at the SNPs written to file):
rootOutFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/kMers/on_hg38/' 
rootOut = rootOutFrq + chromoName + r'/SNP/'
modelFileName = str(k) + '-mer' 
genomeIdName = chromoName
snpIdName = '1000Gs'
snpIndexOffset = 1
predictionArray = predArrayFrq
qualArray = qualArrayFrq
labelArray = corrLabelArray
sampledPositions = samplePositionsFrq
sampledGenomeSequence = sampleGenomeSequence
title = 'k=5 central'
writeOut_b = 0
probsAtSnp_1000G = snp.fetchProbsAtSnps(rootOutput = rootOut, snpInfoArray = snpInfoArray, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b, chrNr = chrNr, genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#get 2d-histo:
saveAtDpi = 300
snp.snpHisto2D(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdName = snpIdName, probsAtSnp = probsAtSnp_1000G, title = title, saveAtDpi = saveAtDpi)


#Get density plot of diff of prob's of ref base and allele (log of) for 1000G, clin-var and cosmics: 
#for the Cosmic and clinVar snp's -- and to get a density of all three in one plot, run the above code
#for the 1000Gs and then the same for the other sets:
fileName = 'clinvar_20200310_filter_INDEL.vcf'
snpInfoArray_Clin = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'ClinVar'
probsAtSnp_clinVar = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_Clin, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)

#and

fileName = 'CosmicNonCodingVariants.vcf'
snpInfoArray_CosmicNC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, non-coding'
probsAtSnp_cosmicNC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicNC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)


#and

fileName = 'CosmicCodingMuts.vcf'
snpInfoArray_CosmicC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, coding'
probsAtSnp_cosmicC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)


#Then call the density plot fct:
snpIdNameList = ['1000Gs', 'Cosmic, non-coding', 'Cosmic, coding', 'ClinVar']
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_cosmicNC, probsAtSnp_cosmicC, probsAtSnp_clinVar ]
log_b = 1
saveAtDpi = 300
snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = 'LSTM1', log_b = log_b, saveAtDpi = saveAtDpi)


#############################
#For the Markov model (k=14):
#############################

import snpAnalysis as snp
#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'
chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'

snpInfoArray = snp.readSNPdata(rootSnp + fileName)

#We first need to get the pred-arrays for the model. For this we first read in the prob's from
#Yuhu's run of AK's model:
#Read in prob's from external: 
fileName = r'/isdata/kroghgrp/wzx205/scratch/01.SNP//03.Bidir_Markov_model/23.CHR22/Chr22.probs'
positionArrayMarkov, predArrayMarkov, refBaseListMarkov = stats.readInProbsFromExternal(fileName)
#As in the k-mer case, we get the pred-arrays from another model's indexing (here a NN-model, as run right above).
#This is id to the wa we got the pred-arrays for the LR test Markov vs LSTM:
samplePositions = samplePositionsNN
getAllPositions_b = 0
sampleGenomeSequence = encodedGenomeData[3] # #from call tostats.dataGen.encodeGenome right above; genomeSeqString is 3rd entry of encodedGenomeData
k = -1 #placeholder
displacementOfExternal = 1 #indexing start from 1 in results from Markov model
predArrayMarkov, qualArrayMarkov, samplePositionsMarkov, sampleGenomeSequenceEncoded = stats.getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence,  labelArray = labelArrayNN, positionsExternal= positionArrayMarkov, probsExternal= predArrayMarkov, refBaseListExternal = refBaseListMarkov,  fileGenome =fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= k, displacementOfExternal = displacementOfExternal)



#Then to get the SNP triangle plot on k-mer model (optionally the prob's at the SNPs written to file):
rootOutMarkov = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/on_hg38/Markovs/Markov14/avgRevCompl/hg38_chr22/' 
rootOut = rootOutMarkov + 'SNP/'
modelFileName = 'Markov14'
genomeIdName = chromoName
snpIdName = '1000Gs'
snpIndexOffset = 1
predictionArray = predArrayMarkov
qualArray = qualArrayMarkov
sampledPositions = samplePositionsMarkov
labelArray = sampleGenomeSequenceEncoded
title = 'Markov k=14'
writeOut_b = 0
probsAtSnp_1000G = snp.fetchProbsAtSnps(rootOutput = rootOut, snpInfoArray = snpInfoArray, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions, startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b, chrNr = chrNr, genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)
#get 2d-histo:
saveAtDpi = 300
snp.snpHisto2D(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdName = snpIdName, probsAtSnp = probsAtSnp_1000G, title = title, saveAtDpi = saveAtDpi)


#Get density plot of diff of prob's of ref base and allele (log of) for 1000G, clin-var and cosmics: 
#for the Cosmic and clinVar snp's -- and to get a density of all three in one plot, run the above code
#for the 1000Gs and then the same for the other sets:
fileName = 'clinvar_20200310_filter_INDEL.vcf'
snpInfoArray_Clin = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'ClinVar'
probsAtSnp_clinVar = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_Clin, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)

#and

fileName = 'CosmicNonCodingVariants.vcf'
snpInfoArray_CosmicNC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, non-coding'
probsAtSnp_cosmicNC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicNC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)


#and

fileName = 'CosmicCodingMuts.vcf'
snpInfoArray_CosmicC = snp.readSNPdata(rootSnp + fileName)
snpIdName = 'Cosmic, coding'
probsAtSnp_cosmicC = snp.fetchProbsAtSnps(rootOutput = rootOut, chrNr = chrNr, snpInfoArray = snpInfoArray_CosmicC, snpIndexOffset = snpIndexOffset, predictionArray = predictionArray, qualArray = qualArray, labelArray = labelArray, sampledPositions = sampledPositions,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, modelFileName = modelFileName, writeOut_b = writeOut_b,  genomeIdName = genomeIdName, snpIdName = snpIdName, title = title)


#Then call the density plot fct:
snpIdNameList = ['1000Gs', 'Cosmic, non-coding', 'Cosmic, coding', 'ClinVar']
probsAtSnpList = [probsAtSnp_1000G, probsAtSnp_cosmicNC, probsAtSnp_cosmicC, probsAtSnp_clinVar ]
log_b = 0
saveAtDpi = 300
snp.snpProbDiffDensity(rootOutput = rootOut, modelFileName = modelFileName, genomeIdName = genomeIdName, snpIdNameList = snpIdNameList, probsAtSnpList = probsAtSnpList, title = 'LSTM1', log_b = log_b, saveAtDpi = saveAtDpi)





_________________________________________________________
    
Part 8.2: handling annotation files (bed files and repeats in genome seq); loading etc:     
_________________________________________________________
    
       
------------------------
Repeats in genomic seq (also fetches length of the chromo seq's --- needed when loading the bed-annotations)
------------------------

#Human
 
#hg38
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']

stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles)


#hg18, chr12 only:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg18/"
rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/hg18/'

chromosomeOrderList = ['hg18_chr12']
stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles)




#Mouse

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
rootOutputAnnoFiles = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"

chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']

stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles)
    
#Zebrafish
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
rootOutputAnnoFiles = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"

chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']

stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles, saveRepeatArray_b =0)


#Drosophila --- only to get and store a dict holding the chromo length we can use this:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/newSplitFct/"
rootOutputAnnoFiles = rootGenome

chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']

saveRepeatArray_b =0
stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles, saveRepeatArray_b = saveRepeatArray_b)


   
#yeast --- only to get and store a dict holding the chromo length we can use this:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
rootOutputAnnoFiles = r"/binf-isilon/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']

saveRepeatArray_b =0
stats.dataGen.generateRepeatArraysGenomeSeqMasking(rootGenome = rootGenome, chromosomeList = chromosomeOrderList, rootOutput = rootOutputAnnoFiles, saveRepeatArray_b = saveRepeatArray_b)

    
------------------------
Bed-file annotations
------------------------

#Human 

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'

    
#Yuhu's folder:
fromYuhusFolder_b = 1
rootAnnotationFiles = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/03.Bidir_Markov_model/"

#Own download of annotation files
#fromYuhusFolder_b = 0
#rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/ucsc/"


chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17e6,1e9], 'hg38_chr14':[16e6,1e9], 'hg38_chr13':[16e6,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[17e6,1e9], 'hg38_chr4':[16e6,1e9], 'hg38_chr3':[16e6,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}

#chromosomeOrderList = ['hg38_chr22]
#chromosomeDict = {'hg38_chr22':[10500000,1e9]}

chromosomeOrderList = [ 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[17e6,1e9], 'hg38_chr4':[16e6,1e9], 'hg38_chr3':[16e6,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}


annotationNameList= ['simple_repeat', 'repeat', 'cds', 'introns', '3UTR', '5UTR', 'gene']

rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'

firstChromoNr = 8

chromoLengthFile = rootOutputAnnoFiles + 'chromosomeLengthDict'

for name in annotationNameList:
    stats.loadAnnotation(rootAnnotationFiles = rootAnnotationFiles, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, chromosomeLengthFile  = chromoLengthFile, firstChromoNr = firstChromoNr, annotationName = name, rootOutput =rootOutputAnnoFiles,  fromYuhusFolder_b = fromYuhusFolder_b)

#establish a combined repeat and simple_repeat annotation (used eg in hg38):
annotationName1 = 'simple_repeat'
annotationName2 = 'repeat'
annotationNameCombined ='RmskAndTrf'
stats.combineAnnotations(chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, chromosomeLengthFile = chromoLengthFile, firstChromoNr = firstChromoNr, annotationName1 = annotationName1, annotationName2 = annotationName2, annotationNameCombined =annotationNameCombined, rootOutput = rootOutputAnnoFiles, organism = 'human')



#----------------------------------------------
#For SINEs and LINEs and their subclasses:
#----------------------------------------------
#I first 'grepped' these in  Yuhu's downloaded bed-files and placed the output
#in /isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/ucsc/ (where I had placed myself). Like so:
grep 'SINE' /isdata/kroghgrp/wzx205/scratch/01.SNP/03.Bidir_Markov_model/21.CHR20/UCSC/chr20_repeat.bed > hg38_chr20_SINE.bed

#Next, just run loadAnnotation wiith organism = other and using the default otherRowIdx1/2 :
organism = 'other'
fromYuhusFolder_b = 0
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/ucsc/"
annotationNameList= ['SINE', 'SINE_Alu', 'SINE_MIR', 'LINE', 'LTR']
chromosomeOrderList = ['hg38_chr20']
chromosomeDict = {'hg38_chr20':[0,1e9]}

rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'

chromoLengthFile = rootOutputAnnoFiles + 'chromosomeLengthDict'
firstChromoNr = 20
extension = '.bed'

buildAnnotationIntervalDict_b = 1

for name in annotationNameList:
    stats.loadAnnotation(rootAnnotationFiles = rootAnnotationFiles, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, chromosomeLengthFile  = chromoLengthFile, firstChromoNr = firstChromoNr, annotationName = name, rootOutput =rootOutputAnnoFiles, buildAnnotationIntervalDict_b = buildAnnotationIntervalDict_b, fromYuhusFolder_b = fromYuhusFolder_b, organism = organism, extension = extension)


#----- 
#Same for  hg18, chr12: first grep these from hg18_chr12_rmsk.txt, eg 
grep 'SINE' hg18_chr12_rmsk.txt > hg18_chr12_rmsk_SINE.txt
grep 'Alu' hg18_chr12_rmsk_SINE.txt > hg18_chr12_rmsk_SINE_Alu.txt

#Next, just run loadAnnotation wiith organism = other and using otherRowIdx1/2  set to 6,7:
organism = 'other'
otherRowIdx1 = 6
otherRowIdx2 = 7
fromYuhusFolder_b = 0
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg18/ucsc/"
annotationNameList= ['SINE', 'SINE_Alu', 'SINE_MIR', 'LINE', 'LTR']
chromosomeOrderList = ['hg18_chr12']
chromosomeDict = {'hg18_chr12':[0,1e9]}

rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/hg18/'

chromoLengthFile = rootOutputAnnoFiles + 'chromosomeLengthDict'
firstChromoNr = 12
extension = '.txt'

buildAnnotationIntervalDict_b = 1

for name in annotationNameList:
    stats.loadAnnotation(rootAnnotationFiles = rootAnnotationFiles, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, chromosomeLengthFile  = chromoLengthFile, firstChromoNr = firstChromoNr, annotationName = name, rootOutput =rootOutputAnnoFiles, buildAnnotationIntervalDict_b = buildAnnotationIntervalDict_b, fromYuhusFolder_b = fromYuhusFolder_b, organism = organism, extension = extension, otherRowIdx1 = otherRowIdx1, otherRowIdx2 = otherRowIdx2)




#Yeast

organism = 'yeast'
#Own download of annotation files
fromYuhusFolder_b = 0
rootAnnotationFiles = r"/binf-isilon/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/ucsc/"

annotationNameList= ['simpleRepeats']

rootOutputAnnoFiles = r"/binf-isilon/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
firstChrNr = 1

chromosomeOrderList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']


for name in annotationNameList:
    stats.loadAnnotation(rootAnnotationFiles = rootAnnotationFiles,  chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, firstChrNr = firstChrNr, annotationName = name, rootOutput =rootOutputAnnoFiles, segmentLength = 1e5, fromYuhusFolder_b = fromYuhusFolder_b, organism = organism)



------------------------------------
#Some CTCF annos on hg18
------------------------------------
#Generate anno files for CTCF+sites based on data downloaded from a paper by Patel et al (?)
pattern = '(chr[X Y 0-9]+):([0-9]+)-([0-9]+)'
inputFileName = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg18/CTCF/ctcf_unoccupied_sites.txt'
outputFileName = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg18/CTCF/CTCF_unoccupied_sites.bed'
stats.bedFileFromCTCFAnno(inputFileName =  inputFileName, outputFileName = outputFileName, pattern = pattern)

#From the generated files, the desired bed files for chr12 can then easily be grepped. The two
#resulting files were sorted with bedtools
# bedtools sort -i hg18_chr12_CTCF_unoccupied_sites.bed > hg18_chr12_CTCF_unoccupied_sites_sorted.bed
#and the sorted files were renamed to hg18_chr12_CTCF_un/occupied.bed.

#The corr annotation arrays can then be obtained using loadAnnotation:
organism = 'other'
otherRowIdx1 = 1
otherRowIdx2 = 2
fromYuhusFolder_b = 0
rootAnnotationFiles = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg18/CTCF/"
annotationNameList= ['CTCF_occupied', 'CTCF_unoccupied']
chromosomeOrderList = ['hg18_chr12']
chromosomeDict = {'hg18_chr12':[0,1e9]}

rootOutputAnnoFiles = '/isdata/kroghgrp/tkj375/data/DNA/human/hg18/'

chromoLengthFile = rootOutputAnnoFiles + 'chromosomeLengthDict'
firstChromoNr = 12
extension = '.bed'

buildAnnotationIntervalDict_b = 1

for name in annotationNameList:
    stats.loadAnnotation(rootAnnotationFiles = rootAnnotationFiles, chromosomeOrderList = chromosomeOrderList, chromosomeDict = chromosomeDict, chromosomeLengthFile  = chromoLengthFile, firstChromoNr = firstChromoNr, annotationName = name, rootOutput =rootOutputAnnoFiles, buildAnnotationIntervalDict_b = buildAnnotationIntervalDict_b, fromYuhusFolder_b = fromYuhusFolder_b, organism = organism, extension = extension, otherRowIdx1 = otherRowIdx1, otherRowIdx2 = otherRowIdx2)


###################################################################################
#Gather some info about input
###################################################################################

#Human
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeOrderList = chromosomeOrderList[::-1]

#for tex-table.
rootOutput = rootGenome
fileName = 'table_chromoInfo_hg38.txt'
captionText = 'Human genome, assembly hg38. Length of the  autosomal chromosomes (nr of bases) and number of different bases in two checks (see text).' 


#Mouse
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']

#for tex-table.
rootOutput = rootGenome
fileName = 'table_chromoInfo_GRCm38.txt'
captionText = 'Mouse genome, assembly GRCm38 (mm10). Length of the  autosomal chromosomes (nr of bases) and number of different bases in two checks (see text).' 
 
    
#Zebrafish
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
chromosomeOrderList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']

rootOutput = rootGenome
fileName = 'table_chromoInfo_GRCz11.txt'
captionText = 'Zebrafish genome, assembly GRCz11. Length of the  autosomal chromosomes (nr of bases).' 



#Drosophila:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/"
chromosomeOrderList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']

rootOutput = rootGenome
fileName = 'table_chromoInfo_r6.18.txt'
captionText = 'Fruit fly genome, assembly r6.18. Length of the  autosomal chromosomes (nr of bases).' 


#Yeast
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
chromosomeOrderList  = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']

#for tex-table.
rootOutput = rootGenome
fileName = 'table_chromoInfo_R64.txt'
captionText = 'Yeast genome, assembly R64. Length of the  autosomal chromosomes (nr of bases) and number of different bases in two checks (see text).' 



#load the chromo-lenght and the two chromo-checks:
import cPickle as pickle
loadFile = rootGenome + r'/chromosomeLengthDict'
chromosomeLengthDict = pickle.load(open( loadFile, "rb"))
loadFile = rootGenome + 'checkChromoSeqs/resultsDict_useFastReadGenome.p'   
checkChromoDict =  pickle.load(open( loadFile, "rb"))
loadFile = rootGenome + 'resultsDict_checkOneHotEncoding.p'
checkOneHotDict =  pickle.load(open( loadFile, "rb"))
    
infoDict = dataGen.gatherGenomeInfo(rootGenome, chromosomeOrderList, chromosomeLengthDict, checkChromoDict, checkOneHotDict)
#make the corr tex-table:
nameChangeDict = {'length':'length', 'diffsChrCheck':'#diff bases chr check', 'diffsOnHotCheck':'#nr diff 1-hot check'}
stats.makeTexTable(inputDict = infoDict, rowNames = chromosomeOrderList, rowColHeading = 'chr', captionText =captionText,  nameChangeDict=nameChangeDict, decPoints = 0, rootOutput = rootOutput, fileName = fileName)


#############################
## a redump

#LSTM1: w/wo avg on rev compl:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"

#LSTM11: as LSTM1 but at earlier training stage:
rootOutput  = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM11/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr15"



#LSTM2: as LSTM1 but not trained w aug rev compl:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM2/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM2/"
modelFileNameNN ="modelLSTM__2Conv2LayerLstm_flanks200_win3_filters64And256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr196"


#LSTM4: flanks 50, trained on hg19
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/LSTM4/"
modelFileNameNN = 'modelLSTM__1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr150'

#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'

#Mouse model (same settings as the human LSTM4) used here for predicting on the human genome (hg38):
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'


chromosomeList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']

segmentLength = 1000000
maxNrSegments = 250 
augmentWithRevComplementary_b = 0
stats.redumpAsSparse(rootOutput = rootOutput, 
                        modelFileName = modelFileNameNN,
                        chromoList = chromosomeList, 
                        segmentLength = segmentLength,
                        maxNrSegments = maxNrSegments,
                        augmentWithRevComplementary_b = augmentWithRevComplementary_b,
                        on_binf_b = 1)



'''

#Un/comment these lines if /no GPU is needed --------------------------------
#to prevent the process from taking up all the ram on the gpu upon start:
#import tensorflow as tf
#
#config = tf.ConfigProto(device_count = {'GPU': 0})
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#
##tf.device('/gpu:1')
#
#sess = tf.Session(config=config)
#tf.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
#
#-------------------------------------

import tensorflow as tf

from tensorflow.keras.models import model_from_json


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

import frqModels as frqM

import dnaNet_dataGen as dataGen
import dnaNet_dataUtils as dataUtils

import scipy.stats as stats

#outcommented if py3
#import cPickle as pickle
import pickle

from scipy.fftpack import fft, ifft  #, dst, rfft, irfft

from scipy.sparse import csr_matrix

import csv

import os

from matplotlib.colors import LogNorm

import re


modelNameChangeDict = {'LSTM1':'LSTM200', 'LSTM4':'LSTM50', 'LSTM4S':'LSTM50S', 'LSTM4P':'LSTM50P', 'LSTM11':'LSTM200early', 'mouseLSTM4':'mouseLSTM50'}

colorMap='Set2'
colorList = cm.get_cmap(colorMap)
colorDict = {'LSTM1':colorList(0), 'LSTM4S':colorList(1), 'LSTM4':colorList(2), 'LSTM4P':colorList(3), 'LSTM11':colorList(4), 'mouseLSTM4':colorList(6) }

###################################################################################
# Predictions, accuracy, Fouriers etc etc
###################################################################################

def predictOnChromosomes(rootGenome, 
                         chromosomeDict,
                         chromosomeOrderList, 
                         rootOutput,
                         rootModel,
                         modelFileName, 
                        segmentLength = 1e6,
                        startAtSegmentDict = {}, 
                        augmentWithRevComplementary_b = 0, #!!!!!
                        customFlankSize = 50,
                        computePredAcc_b = 0, 
                        overlap = 0,
                        leftRight_b = 1, #use 1 for bi-directional models
                        batchSize = 500,
                        windowLength = 5,
                        stepSize = 1,
                        Fourier_b = 0,
                        on_binf_b = 1
                        ):
    '''Aim: to run prediction of a given model on specified strecthes of a list of chromosomes.
    
    chromosomeDict: key: chormosome name, values: [startAtPosition, endAtPosition]
    
    #For human DNA, assembly hg38 (start = startAtPosition):
    chr13: start 16000000
    chr14: start 16000000
    chr15: start 17000000
    chr21: start 5010000
    chr22: start 10500000 
    
    chromosomeOrderList: ust gives the order in which the function processes the 
    chromosomes. The list should only contain chromo names that are keys in chromosomeDict.
    '''
    
    #Dump key input so as to be able to replicate:
    dumpFile = rootOutput + '/chromosomeDict'
    pickle.dump(chromosomeDict, open(dumpFile, "wb"))
    dumpFile = rootOutput + '/startAtSegmentDict'
    pickle.dump(startAtSegmentDict, open(dumpFile, "wb"))
    
    
    for chromoName in chromosomeOrderList:
        
        fileName = chromoName + ".txt"
        fileGenome = rootGenome +fileName
        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100 #droso 200
        startAtPosition, endAtPosition  = chromosomeDict[chromoName]
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b = 1 #!!!
        randomChromo_b = 0
        avoidChromo = []
    
        #encoding the data:
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
                               
                               
        genomeIdName = chromoName + '_seg' + str(int(segmentLength))
        
        startAtSegment = 0
        if chromoName in startAtSegmentDict:
            
            startAtSegment = startAtSegmentDict[chromoName]

        #Create a directory if its not already there:
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):

            os.makedirs(rootOutput_thisChromo)
            print("Directory " + rootOutput_thisChromo + " created. Output will be placed there.")
    
                        
        #Make a log file containing the info needed for the run:
        runDataFileName = rootOutput_thisChromo +  'runData.txt'
        runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
        
        s = "Parameters used in a run of predictOnChromosomes (of module dnaNet_stats)." + "\n"   
        s += 'fileGenome: ' + fileGenome + "\n"
        s += 'chromoName: ' + chromoName + "\n"
        s += 'segmentLength: ' + str(segmentLength) + "\n"
        s += 'Derived genomeIdName then : ' + genomeIdName + "\n"
        s += 'For this chromo start at (startAtSegment): ' + str(startAtSegment) + "\n"
        s += 'rootModel : ' + rootModel + "\n"
        s += 'modelFileName : ' + modelFileName + "\n" 
        runDataFile.write(s)
        
        s = "chromosomeDict and startAtSegmentDict are pickle-dumped in folder above this one (so get them there). They are:" + "\n"
        s += "chromosomeDict: " + str(chromosomeDict)  + "\n" 
        s += "chromosomeOrderList: " + str(chromosomeOrderList)  + "\n" 
        s += "startAtSegmentDict: " + str(startAtSegmentDict)  + "\n" 
        runDataFile.write(s)

        s = "Parameter values used in the call to encodeGenome:" + "\n"
        s += "exonicInfoBinaryFileName: " + exonicInfoBinaryFileName  + "\n"
        s += 'chromoNameBound: ' + str(chromoNameBound) + "\n"
        s += 'startAtPosition: ' + str(startAtPosition) + "\n"
        s += 'endAtPosition: ' + str(endAtPosition) + "\n"
        s += 'outputEncoded_b: ' + str(outputEncoded_b) + "\n"
        s += 'outputEncodedOneHot_b: ' + str(outputEncodedOneHot_b) + "\n"
        s += 'outputEncodedInt_b: ' + str(outputEncodedInt_b) + "\n"
        s += 'outputAsDict_b: ' + str(outputAsDict_b) + "\n"
        s += 'outputGenomeString_b: ' + str(outputGenomeString_b) + "\n"
        s += 'randomChromo_b: ' + str(randomChromo_b) + "\n"
        s += 'avoidChromo: ' + str(avoidChromo) + "\n"
        runDataFile.write(s)



        predictAccrossGenomeInSegments(rootOutput = rootOutput_thisChromo, 
                                             rootModel = rootModel, 
                                             modelFileName = modelFileName, 
                                             genomeIdName = genomeIdName, 
                                             encodedGenomeData = encodedGenomeData, 
                                             augmentWithRevComplementary_b = augmentWithRevComplementary_b, 
                                             segmentLength = segmentLength, 
                                             customFlankSize = customFlankSize,
                                            computePredAcc_b = computePredAcc_b, 
                                            overlap = 0,
                                            startAtSegment = startAtSegment, 
                                            leftRight_b = leftRight_b,
                                            batchSize = batchSize,
                                            windowLength = windowLength,
                                            stepSize = stepSize,
                                            Fourier_b = Fourier_b,
                                            on_binf_b = on_binf_b)  
                                   



def computeAccuracyOnChromosomes(rootGenome, 
                         chromosomeDict,
                         chromosomeOrderList, 
                         rootOutput,
                         rootModel,
                         modelFileName,
                         extension = ".txt",
                        segmentLength = 1e6,
                        averageRevComplementary_b = 0, #!!!!!
                        startAtSegmentDict = {}, 
                        windowLength = 1,
                        stepSize = 1,
                        Fourier_b = 0,
                        defaultAccuracy = 0.25,
                        on_binf_b = 1,
                        forATorGCbias_b = 0,
                        rootOutput_forATorGCbias = '',
                        modelFileName_forATorGCbias = '',                             
                             recodeA = [1,0,0,0],
                             recodeC = [0,1,0,0],
                             recodeG = [0,0,1,0],
                             recodeT = [0,0,0,1],
                        randomPredArray_b = 0,
                        firstPosition_startAtSegment = 0): 
    '''Aim: to compute the accuracy of the predictions of a given model on specified strecthes of a list of chromosomes.
    
    chromosomeDict: key: chormosome name, values: [startAtPosition, endAtPosition]
    
    #For human DNA, assembly hg38 (start = startAtPosition):
    chr13: start 16000000
    chr14: start 16000000
    chr15: start 17000000
    chr21: start 5010000
    chr22: start 1050000 
    
    chromosomeOrderList: ust gives the order in which the function processes the 
    chromosomes. The list should only contain chromo names that are keys in chromosomeDict.
    '''
    
    
    for chromoName in chromosomeOrderList:
        
        fileName = chromoName + extension
        fileGenome = rootGenome + fileName
        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 200 #100
        startAtPosition, endAtPosition  = chromosomeDict[chromoName]
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b = 1 #!!!
        randomChromo_b = 0
        avoidChromo = []
    
        #encoding the data:
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
                               
                               
        genomeIdName = chromoName + '_seg' + str(int(segmentLength))
        
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):
            
            if forATorGCbias_b == 0:
                print ("The appropriate folder (%s) for holding the predArray was not found, so the predArray is probably not run. Remedy and rerun.")
                return
                
            
        if forATorGCbias_b == 1:
                
                print("I'm doing a forATorGCbias run!")
                rootOutput_forATorGCbias_thisChromo = rootOutput_forATorGCbias +  chromoName + '/'
                if not os.path.exists(rootOutput_forATorGCbias_thisChromo):
                    os.makedirs(rootOutput_forATorGCbias_thisChromo)
                    print("Directory " + rootOutput_forATorGCbias_thisChromo + " created. Output will be placed there.")
                
        else:
            
            rootOutput_forATorGCbias_thisChromo = ''
            
            

        startAtSegment = 0
        if chromoName in startAtSegmentDict:
            
            startAtSegment = startAtSegmentDict[chromoName]
            
        computeAccuracyOnSegments(rootOutput = rootOutput_thisChromo,
                             modelFileName = modelFileName , 
                             segmentLength = segmentLength, 
                             genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                             encodedGenomeData = encodedGenomeData,
                             averageRevComplementary_b = averageRevComplementary_b,
                             startAtSegment = startAtSegment, 
                             windowLength = windowLength,
                             stepSize = stepSize,
                             defaultAccuracy = defaultAccuracy,
                             Fourier_b = Fourier_b,
                             forATorGCbias_b = forATorGCbias_b,
                             rootOutput_forATorGCbias = rootOutput_forATorGCbias_thisChromo,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias,
                             recodeA = recodeA,
                             recodeC = recodeC,
                             recodeG = recodeG,
                             recodeT = recodeT,
                             randomPredArray_b = randomPredArray_b,
                             firstPosition_startAtSegment = firstPosition_startAtSegment)

        
        

def predictOnGenomeSamples(rootOutput,
                        rootModel,
                        modelFileName, 
                        outputFromGetAllSamplesFromGenome,
                        genomeIdName, #for saving the prediction array; could eg be a chromo name
                        customFlankSize = 50,
                        computePredAcc_b = 0, 
                        overlap = 0,
                        leftRight_b = 0,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        Fourier_b = 0,
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence or for a set of sample positions (encoded) by 
    using the input estimated/trained model.
    
    Input:
        
        rootOutput: path to dir of output
        rootModel: path to dir of model file 
        modelFileName: name of the model file for the trained model (must have a .h5 companion)
        outputFromGetAllSamplesFromGenome: tuple as returned by dataGen.getAllSamplesFromGenome

        
        computePredAcc_b: boolean; whether or not (0) to compute the accuracy on predicting the base at each position provided. If using 
                          sampled positions not covering a contiguous part of the genome, the windowing does not really makes sense (so set 
                          windowLength = 1 and stepsize = 1)
        
        
    Returns: predArray, Q, S, avgPred where
    
    predArray: the predicted distribution of the four letters at each position in the input genomic data (encodedGenomeData)
    Q: array of booleans indicating for each position whether the sample is qualified or not (letter ACGT or not)
    S: indicator array of the actual genomic positions(1 means that the position was included; this refers to positions in the enocodedGenomeData) 
    avgPred: array of average accuracy obtained by averaging over windows (one average per step)
        
    '''
    
    net = model_from_json(open(rootModel + modelFileName).read())
    net.load_weights(rootModel + modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
#    sizeOutput = net.output_shape[1]

    #Decipher the test data:
    #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
    Xt, Yt, Rt, Qt, St, St_boolean, augWithRevCmpl_b = outputFromGetAllSamplesFromGenome
    
    allDisqualified_b = 0
    if np.sum(Qt) < 1:
        
        allDisqualified_b  = 1
    
    lSampledPositions = len(St)
    print("Nr of sampled positions: %d" % lSampledPositions)
    
    print("Yt shape ", Yt.shape)

    lSamples_X, doubleFlankSize, letterShape = Xt.shape 

    #for some models we have split the left and right flanks in the input the model:
    if leftRight_b == 1:
        
        Xt_lr = np.zeros(shape = (lSamples_X, 2, customFlankSize, letterShape), dtype = 'float32') 
        Xt_l = np.zeros(shape = (lSamples_X, customFlankSize, letterShape), dtype = 'float32') 
        Xt_r = np.zeros(shape = (lSamples_X, customFlankSize, letterShape), dtype = 'float32')                
        
        for i in  range(lSamples_X): 
            Xt_left = Xt[i, :(customFlankSize + overlap) , :].copy()
            Xt_right = Xt[i, (customFlankSize - overlap):, :].copy()
            #and reverse it:
            Xt_right = np.flip(Xt_right, axis = 0)
#            if i == 0:
#                print Xt[i, (customFlankSize - overlap):, :][:5]
#                print Xt_right[-5:]
            #Then
            Xt_lr[i][0] = Xt_left
            Xt_lr[i][1] = Xt_right
            
            Xt_l[i]= Xt_left
            Xt_r[i] = Xt_right
            
        print("Xt_lr shape ", Xt_lr.shape)



    lSamples_Y = len(Yt)
    
    if lSamples_Y != lSamples_X:
        print("The number of sequences does not match the number of labels! --- this is a no-go!!!")
    if lSampledPositions != lSamples_Y:
        if augWithRevCmpl_b == 0:
            print("The number of sampled positions does not match the number of samples! --- this is a no-go!!!")    
        else:
            if 2*lSampledPositions != lSamples_Y:
                print("The number of sampled positions does not corr to the number of samples! --- this is a no-go!!!")    


    #Call the prediction
    if allDisqualified_b == 0:  
             
        if leftRight_b == 0:
            
            predArray = net.predict(Xt, batch_size = batchSize)
        
        elif leftRight_b == 1:
            
            predArray = net.predict([Xt_l, Xt_r], batch_size = batchSize)
        
        print("Nr of samples: %d ; of which are predicted: %d" % (lSamples_X, predArray.shape[0]))
        
        if lSamples_X != predArray.shape[0]:
            print("Nr predictions does not match the number of samples -- this is a no-go!! (implies that positions in prediction array do not corr to genome positions)")
        
        
        #If the set of samples is invariant under reverse complementation we 
        #let the prediction at each position be the "average" of the prediction
        #and the complemented one: 
        #newPred(X) = (pred(i)(X) + pred(idx of rev complemented sample at i)(complement(X))
        #for X = A,C,G,T and where, by construction, idx of rev complemented sample at i = i + length of genome seq
        if augWithRevCmpl_b == 1:
            predArrayNew = np.zeros(shape = (lSampledPositions, letterShape))
            for i in range(lSampledPositions):
                predArrayNew[i] = 0.5*(predArray[i] + predArray[i+lSampledPositions].take([3,2,1,0])) #order is ACGT; so complemented TGCA ie it takes 0123 to 3210 
            predArray = predArrayNew
            labelArray = Yt.take(range(lSampledPositions), axis = 0)
            qualifiedArray = Qt.take(range(lSampledPositions), axis = 0)
        else:
            labelArray= Yt
            qualifiedArray = Qt
            
    else: #allDisqualified_b == 1; in this case just put in a dummy pred-array and keep the others (Yt,Qt .. )
        
        labelArray= Yt
        qualifiedArray = Qt
        predArray = np.zeros(shape = (lSampledPositions, letterShape))
        
        
    #Keep a copy of the results:
    dumpFile = rootOutput + modelFileName + '_' + 'labelArray' + '_' + genomeIdName + '_avgRevCompl' + str(augWithRevCmpl_b)
    pickle.dump(labelArray, open(dumpFile, "wb") )
    dumpFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdName + '_avgRevCompl' + str(augWithRevCmpl_b)
    pickle.dump(predArray, open(dumpFile, "wb") )
    dumpFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdName  + '_avgRevCompl' + str(augWithRevCmpl_b)
    pickle.dump(qualifiedArray, open(dumpFile, "wb") )
    dumpFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdName  + '_avgRevCompl' + str(augWithRevCmpl_b)
    pickle.dump(St, open(dumpFile, "wb") )
    
    #this array is very sparse so let's sparsify it:
    St_boolean = csr_matrix(St_boolean)
    dumpFile = rootOutput + modelFileName + '_' + 'sampledPositionsBoolean' + '_' + genomeIdName  + '_avgRevCompl' + str(augWithRevCmpl_b)
    pickle.dump(St_boolean, open(dumpFile, "wb") )
            
    predReturn = []
    if computePredAcc_b == 1:
        
        predReturn = computeAccuracyOnSamples(rootOutput = rootOutput,
                                              modelFileName = modelFileName, 
                                           genomeIdName = genomeIdName,
                                           averageRevComplementary_b = augWithRevCmpl_b,
                             labelArray = labelArray,
                             repeatInfoArray = Rt,
                             predictionArray = predArray,
                            qualifiedArray = qualifiedArray,
                            windowLength = windowLength,
                        stepSize = stepSize,
                        Fourier_b = Fourier_b)
    
    return predArray, qualifiedArray, predReturn
    
    
def computeAccuracyOnSamples(rootOutput,
                             modelFileName, 
                             genomeIdName,
                             labelArray,
                             repeatInfoArray,
                             predictionArray,
                             averageRevComplementary_b,
                             qualifiedArray,
                             windowLength = 100,
                             stepSize = 100,
                             defaultAccuracy = 0.25,
                             Fourier_b = 0,
                             forATorGCbias_b = 0,
                             rootOutput_forATorGCbias = '',
                             modelFileName_forATorGCbias = '',
                             recodeA = [0,0,0,0],
                             recodeG = [1,1,1,1],
                             recodeC = [1,1,1,1],
                             recodeT = [0,0,0,0]):
    '''
    Fourier_b: whether to include a Fourier transform of the avg-prediction or not; there is separate code
               for Fourier runs in this module, so usually would not be called by this function. 
               
    forATorGCbias_b:  use (set to 1) for the case of computing AT or GCs bias (takes recoding the labels/letters in dnaNet_dataGen too, see usage section). Default: 0 
    rootOutput_forATorGCbias: path to output folder when forATorGCbias_b is set to 1.
    modelFileName_forATorGCbias: model name to use (replaces the true modelFileName) when forATorGCbias_b is set to 1.'''
    
    #Replace modelFileName when code is used for CG or AT bias run. 
    #This just implies that all results will be stored for the replacing model name (eg "ATbias")
    if forATorGCbias_b == 1:
        
        modelFileName = modelFileName_forATorGCbias
        rootOutput= rootOutput_forATorGCbias
        
        #relabel the labeArray:
        recodeA_asArray = np.asarray(recodeA, dtype ='int8')
        recodeC_asArray = np.asarray(recodeC, dtype ='int8')
        recodeG_asArray = np.asarray(recodeG, dtype ='int8')
        recodeT_asArray = np.asarray(recodeT, dtype ='int8')
        
        for i in range(labelArray.shape[0]):
            
            if np.array_equal(labelArray[i], dataGen.codeA_asArray):
                labelArray[i] = recodeA_asArray
            elif np.array_equal(labelArray[i], dataGen.codeG_asArray):
                labelArray[i] = recodeG_asArray
            elif np.array_equal(labelArray[i], dataGen.codeC_asArray):
                labelArray[i] = recodeC_asArray
            elif np.array_equal(labelArray[i], dataGen.codeT_asArray):
                labelArray[i] = recodeT_asArray 


    
    if stepSize > windowLength:
        
        print("stepSize must not be set larger than the windowLength. Correct and rerun.")
        return        
        
    
    lSamples = predictionArray.shape[0]
    
    nrSteps = int((lSamples - windowLength)/stepSize) + 1  #+1 added 8th Jan '20 so runs before that are missing one step (no issue)
    
    print("lSamples ", lSamples,", nrSteps ", nrSteps )
    
    cntTot = 0.0
    cntDisqTot = 0.0
    cntCorr = 0.0
    cntCorrRep = 0.0
    cntTotRep  = 0.0
            
    avgPred = np.zeros(shape = nrSteps, dtype = 'float32')
    
    #default value (will be returned if all positions are disqualifided):
    predReturn = avgPred, cntCorr, cntTot, cntCorrRep, cntTotRep,  []
    
    if np.sum(qualifiedArray) < 1:
        
        #dump the default values:
        dumpFile = rootOutput +  modelFileName + '_predReturn_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
        pickle.dump(predReturn, open(dumpFile,"wb") )
            
        return predReturn
    
    ##############################################################################################
    ## OBS: from here only done if np.sum(qualifiedArray) >= 1
    ##############################################################################################
    
    Y = labelArray
    R = repeatInfoArray
    
    #We divide in two cases: if the windowLength = 1 (= stepSize)     
    if windowLength == 1: #here nrSteps = lSamples
        
        corr = 0.0
        cntErr = 0.0
        corrDisq = 0.0
        qualified_b = 1
        for i in range(lSamples):
            
            corr = 0
            qualified_b = 1 #to record whether the window is qualified or not (ie contains a non-ACGt letter)
            #Check if all positions in window are qualified:
            if qualifiedArray[i] != 1:
                qualified_b = 0
                cntDisqTot += 1.0
                print("window disqualified at sample ", i)
                
            predIdx = np.argmax(predictionArray[i])
                
            if Y[i][predIdx] > 0.5 and Y[i][predIdx] < 2: # last < 2 part just to bee on the safe side (and could be avoided if we had encoded the W's smarter ... :--)
                cntCorr += 1.0
                if qualified_b == 0:
                    corrDisq += 1.0
                if R[i] > 0.5:
                    cntCorrRep += 1.0
                    cntTotRep  += 1.0
                corr = 1
                
            else:
                corr = 0
                if R[i] > 0.5:
                    cntTotRep  += 1.0
                
            cntTot += 1
            
            if qualified_b  == 1:
                avgPred[i] = corr
            else:
                avgPred[i] = defaultAccuracy
            

    else: #windowLength != 1
    
        windowList = []
        windowArray = np.zeros(shape = windowLength, dtype = 'float32')
        cntErr = 0
        qualified_b = 1
        for j in range(nrSteps):
            
            #first window: read in the following windowLength worth of sites:
            if j == 0:
                
                qualified_b = 1 #to record whether the window is qualified or not (ie contains a non-ACGt letter)
                #Check if all positions in window are qualified:
                if np.sum(qualifiedArray[:windowLength]) != windowLength:
                    qualified_b = 0
                    print("window disqualified at step ", j)
                
                for i in range(windowLength): #range(predArray.shape[0]):
                    
    #                print " ".join(map(str,Yt[i])), " ".join(map(str,predArray[i]))
    
                    #Check if position is qualified; if not skip the sample:
                    if qualifiedArray[i] == 0:
    #                    qualified_b = 0
                        windowList.append(0.5)
                        continue
                    
                    predIdx = np.argmax(predictionArray[i])
                    
    #                print Y[i], predIdx
                    
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
                    avgPred[j] = defaultAccuracy
                
            else:
                
                qualified_b = 1 #to record whether the window is qualified or not (ie contains a non-ACGt letter)
                #Check if all positions in window are qualified:
                if np.sum(qualifiedArray[j*stepSize:(j*stepSize + windowLength)]) != windowLength:
                    qualified_b = 0
                    print("window disqualified at step ", j)
                    
                #remove first stepSize elt's from list
                for k in range(stepSize):
                    try:
                        windowList.pop(0)
                    except IndexError:
                        print(j)
                        cntErr += 1
                        
                #Append the stepSize next elts:
                for l in range(stepSize): 
                    i = windowLength + (j-1)*stepSize + l
                    
                    #Check if position is qualified; if not skip the sample:
                    if qualifiedArray[i] == 0:
    #                    qualified_b = 0
                        windowList.append(0.5)
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
                    avgPred[j] = defaultAccuracy
                
            
    plt.figure()       
    plt.plot(avgPred) 
    plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_predPlot.pdf' )    
    plt.close()
    
    
    if Fourier_b == 1:
        
        #Fourier transform it:
        fftAvgPred = rfft(avgPred) #scipy fast Fourier transform
        print(fftAvgPred.shape[0])
        plt.figure()
        plt.title('fft avg prediction')  
        start = 0 #int(nrSteps/34)
        end = nrSteps - 1 #fftAvgPred.shape[0] -1 #int(nrSteps/33)
#        plt.bar(range(start,end),fftAvgPred[start:end]) 
        plt.bar(range(start,end), fftAvgPred[range(start,end)]) 
        #plt.plot(fftAvgPred)
        plt.savefig(rootOutput + modelFileName + '_FourierTransformPredPlot' + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
        plt.close()
            
        args = np.argsort(fftAvgPred)
#        print("Bottom 40 frq's (neg coeffs probably) ", args[:40])
#        print(".. and their coeff's", fftAvgPred[args][:40])
        plt.figure()
        plt.title('fft, frqs vs coeffs, lowest 1000 coeffs')  
        plt.scatter(args[:1000], fftAvgPred[args][:1000])
        plt.savefig(rootOutput + modelFileName + '_FourierTransformPredPlot_lowestCoeffs' + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
#        print "frqs, bottom ", args[:500]
        plt.close()
        
#        print("Top 40 frq's (pos coeffs probably) ", args[::-1][:40])
#        print(".. and their coeff's", fftAvgPred[args[::-1]][:40])
        plt.figure()
        plt.title('fft, frqs vs coeffs, highest 1000 coeffs')  
        plt.scatter(args[::-1][1:1000], fftAvgPred[args[::-1]][1:1000])
        plt.savefig(rootOutput + modelFileName + '_FourierTransformPredPlot_highestCoeffs' + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
#        print "frqs, top ", args[::-1][1:500]
        plt.close()
        
        
    avgCorr = cntCorr/cntTot
    print("cntCorr: %d , cntTot: %d ; average prediction acc : %f" % (cntCorr, cntTot, avgCorr))
    
    avgCorrNoDisq= (cntCorr - corrDisq)/(cntTot -cntDisqTot)
    print("corrDisq: %d , cntDisqTot: %d ; average prediction acc no disq: %f" % (corrDisq,cntDisqTot,avgCorrNoDisq))

    nrReps = np.sum(R)
    if nrReps > 0.5: #ie if there are any repeats recorded
        avgCorrRep = cntCorrRep/cntTotRep
        print("cntCorrRep: %d , cntTotRep: %d" % (cntCorrRep, cntTotRep))
        avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
        print("Average prediction acc at repeats: %f and elsewhere: %f" % (avgCorrRep, avgCorrNonRep))
    else:
        print("No repeat sections were recorded in the genome data.")
    
    if Fourier_b == 1:
        
        predReturn = avgPred, cntCorr, cntTot, cntCorrRep, cntTotRep, args
                
    else:
        
        predReturn = avgPred, cntCorr, cntTot, cntCorrRep, cntTotRep,  []
        
    #dump results:
    dumpFile = rootOutput +  modelFileName + '_predReturn_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
    pickle.dump(predReturn, open(dumpFile,"wb") )
        
    
    return predReturn
    
    
    
def predictAccrossGenomeInSegments(rootOutput, 
                                   rootModel, 
                        modelFileName,
                        genomeIdName, #for saving the prediction array; could eg be a chromo name
                        segmentLength,
                        encodedGenomeData,
                        outputEncodedOneHot_b = 1, 
                        outputEncodedInt_b = 0,
                        augmentWithRevComplementary_b = 0,
                        customFlankSize = 50,
                        startAtSegment = 0, 
                        computePredAcc_b = 0, 
                        overlap = 0,
                        leftRight_b = 0,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        Fourier_b = 0,
                        on_binf_b = 1):
    '''
    Computes the prediction acc to the loaded model across a genome/chr in chunks,
    by segmenting the genome sequence.
    '''

    if len(encodedGenomeData) != 5:
        
        print("You have probably run encodeGenome (dnaNet_dataGen module) without asking to have the genome sequence included in output. Change and rerun.")
    
    else:
        
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString, chromoList =  encodedGenomeData 
      
    genSamplesAtRandom_b = 0 #!!!
    cutDownNrSamplesTo = 1e26
    
    #use these default settings:
    labelsCodetype = 0
    outputEncodedType = 'int8'
    convertToPict_b = 0
    shuffle_b = 0
    inner_b = 1
    shuffleLength = 5
    
    lGenomeSeq = len(genomeSeq)
    
#    #Determine the nr of segments: we want to disregard heading/trailing N's; these
#    #are truncated off; then find the largest nr of segments fitting (adjacent) 
#    #fitting in the truncated genome seq; the remainder is handled separately: 
#    firstNonN = 0
#    lastNonN = lGenomeSeq - 1
#    while ... == 'N':
#        firstNonN += 1
#    while ... == 'N':
#        lastNonN -= 1
    
#    nrSegments = np.floor(float(lastNonN -firstNonN)/segmentLength)

    nrSegments = int(np.floor(float(lGenomeSeq)/segmentLength))   
    
    print("nrSegments: ", nrSegments)
    
    cntTot = 0
    cntCorr = 0.0
    cntCorrRep = 0.0
    cntTotRep  = 0.0
    
    sampledPositionsFirst = 0
    sampledPositionsLast = 0
    for i in range(startAtSegment, nrSegments):
        
        print("Now at segment ", i)
        
        genomeIdNameSeg = genomeIdName + '_segment' + str(i)
    
        startPosition = i*segmentLength
        endPosition = startPosition + segmentLength
    
        outputSamples = dataGen.getAllSamplesFromGenome(encodedGenomeData = encodedGenomeData, startPosition = startPosition, endPosition = endPosition, genSamplesAtRandom_b = genSamplesAtRandom_b, cutDownNrSamplesTo = cutDownNrSamplesTo, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
        X, Y, R, Q, sampledPositions, sampledPositionsBoolean, augWithRevCmpl_b =  outputSamples       
        
        sampledPositionsFirst = sampledPositions[0]
        #check if segments are adjacent
        if i > 0:
             
            if sampledPositionsLast != sampledPositionsFirst -1:
                
                print("Something may be rotten --- the segments are not adjacent! -- but might be ok if you've started at segment nr above 0")
                
        sampledPositionsLast = sampledPositions[len(sampledPositions) -1]
        
        #Some genome seq's/chromo's start with a high number of non-ACGT letters (N's) and/or end wth
        #a long seq of non-ACGTs; if the segment consists entirely of such non-qualified base (positions)
        #we let the prediction arrays be trivial; else we run the prediction:
#        allNs_b = 0
        if np.sum(Q) == 0:
            
            print("All N's in this segment! --- no prediction accucracy is computed")        
#            allNs_b =  1
            #This call will simply store dummy pred-arrays
            predArray, qualifiedArray, predReturn = predictOnGenomeSamples(rootOutput = rootOutput, rootModel = rootModel,  modelFileName = modelFileName, genomeIdName = genomeIdNameSeg, computePredAcc_b = computePredAcc_b, outputFromGetAllSamplesFromGenome = outputSamples, customFlankSize = customFlankSize, leftRight_b = leftRight_b, windowLength = windowLength, stepSize = stepSize, batchSize = batchSize, Fourier_b = Fourier_b)

                    
        else: 
            
            predArray, qualifiedArray, predReturn = predictOnGenomeSamples(rootOutput = rootOutput, rootModel = rootModel,  modelFileName = modelFileName, genomeIdName = genomeIdNameSeg, computePredAcc_b = computePredAcc_b, outputFromGetAllSamplesFromGenome = outputSamples, customFlankSize = customFlankSize, leftRight_b = leftRight_b, windowLength = windowLength, stepSize = stepSize, batchSize = batchSize, Fourier_b = Fourier_b)
    
            avgPred_seg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = predReturn
            
            cntCorr += cntCorr_seg
            cntTot += cntTot_seg
            avgCorr = cntCorr/cntTot
            print("Average prediction acc after segment %d: %f" % (i, avgCorr))
        
            cntCorrRep += cntCorrRep_seg
            cntTotRep += cntTotRep_seg 
            if cntTotRep > 0:
                avgCorrRep = cntCorrRep/cntTotRep
                avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
                print("Average prediction acc at repeats after segment %d: %f and elsewhere: %f" % (i, avgCorrRep, avgCorrNonRep))
        

def assemblePredictArrayFromSegments(rootOutput, 
                        modelFileName,
                        segmentLength,
                        genomeIdName, #for saving the prediction array; could eg be a chromo name
                        nrSegments, #encodedGenomeData,
                        augmentWithRevComplementary_b = 0,
                        on_binf_b = 1,
                        for_GCbias_b = 0,
                        rootGCbias = '', 
                        windowLength = 1,
                        stepSize = 1):
    '''Assembles a complete prediction array from prediction in segments obtained
    by predictAccrossGenomeInSegments.
    
    nrSegments: if you want to assemble all exisiting segments, set this number to the highest segment number or higher
    
    rootOutput: the path to the directory in which the prediction results on segments are kept (output from
    predictAccrossGenomeInSegments)
    

    '''
    
    # if len(encodedGenomeData) != 5:
        
    #     print("You have probably run encodeGenome (dnaNet_dataGen module) without asking to have the genome sequence included in output. Change and rerun.")
    
    # else:
        
    #     genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString, chromoList =  encodedGenomeData 
#
#    
#    lGenomeSeq = len(genomeSeq)
#    
#    nrSegments = int(np.floor(float(lGenomeSeq)/segmentLength))   
#    
#    print "nrSegments: ", nrSegments
        
    #loop thorugh seg's, fetch the stored pred-array for each and update the predArray with it    
    for i in range(nrSegments):        
        
        try:
            
            genomeIdNameSeg = genomeIdName + '_seg' + str(int(segmentLength)) + '_segment' + str(i)
            
            loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(augmentWithRevComplementary_b)
            print(loadFile[:100])
            print(loadFile[100:])
            predArraySeg = pickle.load(open( loadFile, "rb"))
            if i == 0:
                predArray = predArraySeg
            else:
                predArray = np.concatenate((predArray, predArraySeg), axis = 0)
                
            loadFile = rootOutput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(augmentWithRevComplementary_b)
            print(loadFile)
            labelArraySeg = pickle.load(open( loadFile, "rb"))
            if i == 0:
                labelArray = labelArraySeg
            else:
                labelArray = np.concatenate((labelArray, labelArraySeg), axis = 0)
    
            loadFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg  + '_avgRevCompl' + str(augmentWithRevComplementary_b)
            qualArraySeg = pickle.load(open( loadFile, "rb" ) )
            if i == 0:
                qualArray = qualArraySeg
            else:
                qualArray = np.concatenate((qualArray, qualArraySeg), axis = 0)
    
            loadFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(augmentWithRevComplementary_b)
            sampledPositionsSeg = pickle.load(open( loadFile, "rb" ) )
            sampledPositionsSeg.astype('int64')
            if i == 0:
                sampledPositions = sampledPositionsSeg
            else:
                sampledPositions = np.concatenate((sampledPositions, sampledPositionsSeg), axis = 0)
    
            sampledPositionsBoolean = 0 #just a dummy value;this array turned out not to be needed
#            loadFile = rootOutput + modelFileName + '_' + 'sampledPositionsBoolean' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(augmentWithRevComplementary_b)
#            sampledPositionsBooleanSeg = pickle.load(open( loadFile, "rb" ) )
#            #By construction sampledPositionsBooleanSeg contains 0/1 at all positions in genomeSeq
#            #with 1 where the position was sampled (for the considered segment) 
#            if i == 0:
#                sampledPositionsBoolean = sampledPositionsBooleanSeg
#            else:
#                sampledPositionsBoolean += sampledPositionsBooleanSeg
            
            if for_GCbias_b == 1:
                
                loadFile = rootGCbias + 'GCbias_predReturn_'  + genomeIdNameSeg + '_avgRevCompl' + str(augmentWithRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)                
                print(loadFile[:100])
                print(loadFile[100:])
                predReturnSegTot = pickle.load(open( loadFile, "rb"))
                predReturnSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg  = predReturnSegTot
                if i == 0:
                    predReturn = predReturnSeg
                else:
                    predReturn = np.concatenate((predReturn, predReturnSeg), axis = 0)
                    
                
        except IOError:
            print("File %s not found" % loadFile)
            break
        
    if for_GCbias_b == 1:
        
        return predReturn, labelArray, qualArray, sampledPositions, sampledPositionsBoolean
        
    else:
    
        return predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean
    
    

def asemblePredictArrayOnChromosomes(toBeDone):    
                            
    '''Analogous to PredictArrayAcrossChromosomes/predictAccrossGenomeInSegments; this just assembles 
    prediction arrays had for a set chromosomes (in segments more or less coveringeach chromosome)
    So simply loops over the chromos and applies assemblePredictArrayFromSegments to each.'''
    
    pass
    
    

def computeAccuracyOnSegments(rootOutput,
                             modelFileName, 
                             segmentLength,
                             genomeIdName, #for saving the prediction array; could eg be a chromo name
                             encodedGenomeData,
                             averageRevComplementary_b,
                             startAtSegment = 0,
                             windowLength = 100,
                             stepSize = 100,
                             defaultAccuracy = 0.25,
                             Fourier_b = 0,
                             forATorGCbias_b = 0,
                             rootOutput_forATorGCbias = '',
                             modelFileName_forATorGCbias = '',                             
                             recodeA = [1,0,0,0],
                             recodeC = [0,1,0,0],
                             recodeG = [0,0,1,0],
                             recodeT = [0,0,0,1],
                             randomPredArray_b = 0,
                             firstPosition_startAtSegment = 0):

    '''
    Computes accuracy on arrays of predictions in segments obtained by predictAccrossGenomeInSegments. 
     
    rootOutput: path to dir  where prediction results are kept; also used for the output of this function.
    encodedGenomeData: must be 

    '''
    
    if len(encodedGenomeData) != 5:
        
        print("You have probably run encodeGenome (dnaNet_dataGen module) without asking to have the genome sequence included in output. Change and rerun.")
    
    else:
        
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString, chromoList =  encodedGenomeData 

    
    lGenomeSeq = len(genomeSeq)
    
    nrSegments = int(np.floor(float(lGenomeSeq)/segmentLength))   
    
    print("nrSegments: ", nrSegments)
    
#    lGenomeSeqCovered = nrSegments*segmentLength
    
#    predArray = np.zeros(shape = (lGenomeSeqCovered, 4))
#    qualArray = np.zeros(shape = lGenomeSeqCovered)
#    sampledPositionsBoolean = np.zeros(shape = lGenomeSeqCovered)

    #loop thorugh seg's, fetch the stored pred-array for each and update the predArray with it
    for i in range(startAtSegment, nrSegments):        
        
        genomeIdNameSeg = genomeIdName + '_segment' + str(i)
        
        loadFile = rootOutput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        print(loadFile)
        labelArraySeg = pickle.load(open( loadFile, "rb"))
    
        loadFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        qualArraySeg = pickle.load(open( loadFile, "rb" ) )
        
        if randomPredArray_b == 0: #standard case
            loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            print(loadFile)
            predArraySeg = pickle.load(open( loadFile, "rb"))
        elif randomPredArray_b == 1: #only for the case where a random pred array is wanted (eg a dummy pred array is needed or for tessting)
            predArraySeg = labelArraySeg.copy()
            #shuffle it:
            shuffledIdxs = np.arange(labelArraySeg.shape[0])
            np.random.shuffle(shuffledIdxs)
            predArraySeg = predArraySeg.take(shuffledIdxs)
            
            
        #the sampledPositions are used only for fetching the repeat sequence; if the file is not found we generate the sampled positions
        #from the set flanksize; this may be a little unsafe, if you don't know the actual start position of the first segment, so the  
        #results on the repeat-stuff should only be taken serious if you are certain about the start. If the sampledPositions file is found
        #there is of course no issue.
        try:
            loadFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            sampledPositionsSeg = pickle.load(open( loadFile, "rb" ) ) 
    
            sampledPositionsSeg = sampledPositionsSeg.astype(int) #an "unsafe" recasting but should be fine here ...
        except IOError: #compute the sampled Positions:
            
            if i == startAtSegment:
                startPosition = firstPosition_startAtSegment
                stopPosition = startPosition + len(labelArraySeg)
            else:
                startPosition = stopPosition #don't add 1: np,arange includes start, but excludes stop                
                stopPosition = startPosition + len(labelArraySeg)
            sampledPositionsSeg = np.arange(start =startPosition, stop = stopPosition,  dtype = 'int64' )
            
        
        Rt = repeatInfoSeq.take(sampledPositionsSeg, axis = 0)

        predReturn = computeAccuracyOnSamples(rootOutput = rootOutput,
                                              modelFileName = modelFileName, 
                                           genomeIdName = genomeIdNameSeg,
                                           averageRevComplementary_b = averageRevComplementary_b,
                             labelArray = labelArraySeg,
                             repeatInfoArray = Rt,
                             predictionArray = predArraySeg,
                            qualifiedArray = qualArraySeg,
                            windowLength = windowLength,
                        stepSize = stepSize,
                        defaultAccuracy = defaultAccuracy,
                        Fourier_b = Fourier_b, 
                        forATorGCbias_b = forATorGCbias_b,
                        rootOutput_forATorGCbias =rootOutput_forATorGCbias,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias,
                             recodeA = recodeA,
                             recodeC = recodeC,
                             recodeG = recodeG,
                             recodeT = recodeT)
        
            
def getAccuracyOnSegments(rootOutput,
                             modelFileName, 
                             segmentLength,
                             genomeIdName, #for saving the prediction array; could eg be a chromo name
                             averageRevComplementary_b,                              
                             windowLength = 1,
                             stepSize = 1,
                             annotationDict = {},
                             startAtPosition = 0,
                             rootOutput_predReturn = '',
                             modelFileName_predReturn = ''
                             ):
    '''Derives the accuracy from the predReturn arrays obtained by computeAccuracyOnSamples (also
    called by computeAccuracyOnSegments). The predReturn arrays are binary, 1 where the model was
    right, and 0 elsewhere (or default prob where the position was disqualified).


    annotationDict: dictionary mapping annotation types (eg "repeat") to a binary array providing
    the positions where that annotation is had in the (a) genome sequence (the array must of course corr
    to the predictions used -- as def'ed by rootOutput and modelFilename).     
    
    startAtPosition: providing the start position in the genome sequence (genomeIdName) at which the
    predictions start. This info must be id to what was provided when the predictions were done. 
    
    Structure of the code is as in FourierOnSegments.'''
    
    #Step1: seek/load the first return array
    #Step2: while loop; compute acc on currently loaded arrays; then load new and redo until while breaks
    
    #Step1:
    data_b = 0
    i = 0
    
    
    corrAllSegs = 0
    totAllSegs = 0
    
    resultsDictAggr = {}
    
    resultsDict = {}
    resultsDict['all'] = {}
    for annoType in annotationDict:

        resultsDict[annoType] = {}    
    
    
    while data_b == 0:
        
        genomeIdNameSeg = genomeIdName + '_segment' + str(i)
            
        loadFile = rootOutput_predReturn +  modelFileName_predReturn + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
        
#        loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)        
        
        print(loadFile)
            
        try: 
            
            storedPredThisSeg = pickle.load(open(loadFile,"rb"))
    
            avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg
#            avgPredSeg = storedPredThisSeg
            print(i, avgPredSeg.shape)
#            print(avgPredSeg[100000:101000])
#            raw_input("S er den dejligste")
        
            
            #get the qualified array too:
            loadFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            qualArraySeg = pickle.load(open( loadFile, "rb" ) )
            print(qualArraySeg.size)
            
            #get the smapledPositions array too:
            loadFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            positionsArraySeg = pickle.load(open( loadFile, "rb" ) )
            print(positionsArraySeg.size)
            
            data_b = 1
        
#    return avgPredSeg, cntCorr_seg, cntTot_seg
        
        except IOError:
            i += 1
            if i > 1000:
                print("Files not found, last tried: ", loadFile)
                return
        except EOFError:
            i += 1
            if i > 1000:
                print("Empty file, last tried: ", loadFile)
                return
            continue
    
    
    #Step2:
    cnt = 0
    corr = 0
    while data_b == 1:
        
        #Compute the acc: a position counts iff it is qualified and the model is right:
        corr = np.dot(qualArraySeg, avgPredSeg)
        tot = np.sum(qualArraySeg)
        acc = float(corr)/(tot +1) #+1 to avoid ...
        print("Acc at %s: %f based on corr %d and tot %d" % (genomeIdNameSeg, acc, corr, tot))
        corrAllSegs += corr
        totAllSegs += tot
        resultsDict['all'][i] = [acc, corr, tot]
        
        #to compare with:
        acc = float(cntCorr_seg)/(cntTot_seg +1) #+1 to ...
        print("Acc at %s as recorded/done in computeAcc: %f , based on nr of corr %d, tot cnt %d" % (genomeIdNameSeg, acc,  cntCorr_seg, cntTot_seg))
        cntCorrRep_seg, cntTotRep_seg
        
        for annoType in annotationDict:
            
            positionsThisSegment = positionsArraySeg + startAtPosition
            positionsThisSegment = positionsThisSegment.astype(np.int64, casting='unsafe', copy=True)
            print(positionsThisSegment)
            annoArray = annotationDict[annoType].take(positionsThisSegment)
            
            #The annoArray is then an array of 0/1's indicating covering exactly the positions sampled
            #for the (predictions of) current segment. To get the acc for the annoType is then simple:
            #First take the position-wise product of the quals's and the anno --- all positions having
            #a 1 in the resulting array must be considered when estimating the acc:
            qualArrayThisAnno = np.multiply(qualArraySeg, annoArray)
            
            #Then take the dot-with the qual-array etc as above:
            corrAnno = np.dot(qualArrayThisAnno, avgPredSeg)
            totAnno = np.sum(qualArrayThisAnno)
            accAnno = float(corrAnno)/(totAnno +1) #+1 just to avoid div by 0
            print("Acc at %s for annotation %s is: %f based on %d of a total of %d" % (genomeIdNameSeg, annoType, accAnno, corrAnno, totAnno))
            resultsDict[annoType][i] = [accAnno, corrAnno, totAnno]
       
        accRep = float(cntCorrRep_seg)/(cntTotRep_seg +1) #+1 ...
        print("Acc at %s for repeats as recorded/done in computeAcc: %f , based on nr of corr %d, tot cnt %d" % (genomeIdNameSeg, accRep, cntCorrRep_seg, cntTotRep_seg))
                
        #Load data/arrays for the next segment, if any:         
        i += 1
        genomeIdNameSeg = genomeIdName + '_segment' + str(i)

        loadFile = rootOutput_predReturn +  modelFileName_predReturn + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
        

#        loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)        
        
        print(loadFile) 
        try:
            storedPredThisSeg = pickle.load(open(loadFile,"rb"))

            avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg
#            avgPredSeg = storedPredThisSeg
            print(i, avgPredSeg.shape)
#            print(avgPredSeg[100000:101000])
#            raw_input("S er den dejligste")
            
            
            #get the qualified array too:
            loadFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            qualArraySeg = pickle.load(open( loadFile, "rb" ) )
            print(qualArraySeg.size)
            
            #get the sampledPositions array too:
            loadFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            positionsArraySeg = pickle.load(open( loadFile, "rb" ) )
            print(positionsArraySeg.size)          

            data_b = 1
        
        except IOError:
            
            #Compute the acc on all segments:
            #The last seg nr for which data is had is i -1, so range(i) is just right: 
            for annoKey in resultsDict:
                
                corrAllSeg = 0
                totAllSeg = 0
                for j in range(i):
                    
                    accSeg, corrSeg, totSeg = resultsDict[annoKey][j]
                    corrAllSeg += corrSeg
                    totAllSeg += totSeg

                resultsDictAggr[annoKey] = float(corrAllSeg)/totAllSeg, corrAllSeg, totAllSeg 
                
            #dump the results
            dumpFile = rootOutput_predReturn + 'accuracyChromoByAnnoSegDictionary'
            pickle.dump(resultsDict, open(dumpFile, "wb") )      

            dumpFile = rootOutput_predReturn +'accuracyChromoByAnnoDictionary'
            pickle.dump(resultsDictAggr, open(dumpFile, "wb") )      

            return resultsDict, resultsDictAggr 
        
        cnt += 1
    

    
    return resultsDict, resultsDictAggr 
                                  


def getAccuracyChromosomes(chromosomeOrderList, 
                         rootOutput,
                         modelFileName, 
                         segmentLength,
                         averageRevComplementary_b,
                         windowLength = 1,
                         stepSize = 1, 
                         annotationTypes = [],
                         rootAnnotationFiles = '',
                         chromosomeDict = {},
                         rootOutput_predReturn = '',
                         modelFileName_predReturn = ''
                         ):
    '''Compute the accuracy on chromos by applying
    getAccuracyOnSegments to the data of each chromosome.
    
    annotationDict: dictionary mapping annotation types (eg "repeat") to a binary array providing
    the positions where that annotation is had in the (a) genome sequence (the array must of course corr
    to the predictions used -- as def'ed by rootOutput and modelFilename).     
    
    chromosomeDict: dict mapping chromosome-keys to the start/end position in the genome sequence at which the
    predictions start. Should be id to the similar info used when the predictions were done. The chromosome-keys
    must cover the names in the chromosomeOrderList.'''
    
    #for GC/AT bias the rootOutput and modelFileName to be used for fetching the pred-returns (eg 'GCbias') are 
    #diff from the rootOutput/modelFileName used for fetching the qualArrays etc; in case they are not supplied set 
    #them equal:
    if len(rootOutput_predReturn) == 0 or len(modelFileName_predReturn) == 0:
        
        rootOutput_predReturn = rootOutput
        modelFileName_predReturn = modelFileName
    
    
    resultsDictByAnnoSeg = {}
    resultsDictByAnno = {}
    for chromoName in chromosomeOrderList:
        
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        rootOutput_predReturn_thisChromo =  rootOutput_predReturn +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):

            print("The appropriate folder (%s) for holding the predArray was not found, so the predArray is probably not run. Remedy and rerun." % rootOutput_thisChromo)
            return
        
        if len(rootOutput_predReturn) > 0:
            
            if not os.path.exists(rootOutput_predReturn_thisChromo):

                print("The appropriate folder (%s) for holding the predArray was not found, so the predArray is probably not run. Remedy and rerun." % rootOutput_predReturn_thisChromo)
                return
        

        genomeIdName = chromoName + '_seg' + str(int(segmentLength))
        
        startAtPosition, endAtPosition = chromosomeDict[chromoName]
        
        #load annotationArray if there:
        annotationDict = {}
        for annoType in annotationTypes:
            
            annotationFile = rootAnnotationFiles + chromoName + '_annotationArray_' + annoType
            
            print(annotationFile)
        
            
            try:
            
                annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))
                
            except IOError:
                
                print("Annotation file %s not found" %  annotationFile)
                continue
            
        #Just for checking the shapes:
        for annoType in annotationDict:
            print(annoType, annotationDict[annoType].shape)
    
        
#        avgPredSeg, cntCorr_seg, cntTot_seg = computeFourierOnSegments(rootOutput = rootOutput_chr,
        resultsDictByAnnoSeg[chromoName], resultsDictByAnno[chromoName] = getAccuracyOnSegments(rootOutput = rootOutput_thisChromo,
                                     modelFileName = modelFileName,  
                                     segmentLength = segmentLength,
                                     genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                                     averageRevComplementary_b = averageRevComplementary_b,
                                     windowLength = windowLength,
                                     stepSize = stepSize,
                                     annotationDict = annotationDict,
                                     startAtPosition = startAtPosition,
                                     rootOutput_predReturn = rootOutput_predReturn_thisChromo,
                                     modelFileName_predReturn = modelFileName_predReturn)
                                     
    #dump the results; replace/extend existing results:
    try: #load exsiting and replace/extend results
        
        loadFile = rootOutput_predReturn +'accuracyByChromoAnnoSegDictionary'        
        accuracyByChromoAnnoSegDictionary = pickle.load(open(loadFile, "wb") )
        for chromoName in resultsDictByAnnoSeg:
                accuracyByChromoAnnoSegDictionary[chromoName] = resultsDictByAnnoSeg[chromoName]
        
        loadFile = rootOutput_predReturn +'accuracyByChromoAnnoDictionary'
        accuracyByChromoAnnoDictionary = pickle.load(open(loadFile, "wb") )
        for chromoName in accuracyByChromoAnnoDictionary:
            accuracyByChromoAnnoDictionary[chromoName] = resultsDictByAnno[chromoName]
            
        return accuracyByChromoAnnoSegDictionary, accuracyByChromoAnnoDictionary
        
    except IOError: #simply dump the results
                                     
        dumpFile = rootOutput +'accuracyByChromoAnnoSegDictionary'
        pickle.dump(resultsDictByAnnoSeg, open(dumpFile, "wb") )
    
        dumpFile = rootOutput +'accuracyByChromoAnnoDictionary'
        pickle.dump(resultsDictByAnno, open(dumpFile, "wb") )                                     
                     
        return resultsDictByAnnoSeg, resultsDictByAnno
                      


def calculateAggrAccOnChromos(rootOutput, chromosomeDict, dictionaryName = 'accuracyChromoByAnnoDictionary'):
    '''Computes the acc accross all chromos for each annotation type, as covered by the input
    dictionary (loaded from a pickle-dumped dict) for the chromos in the chromosomeDict.'''


    resultsDict = {}
    for chromoName in chromosomeDict:
        
            rootOutputThisChromo = rootOutput + chromoName + '/'
            loadFile = rootOutputThisChromo + dictionaryName
            try:
                
                resultsDictByAnno  = pickle.load(open(loadFile,"rb"))
                
                if not(chromoName in resultsDict):
                    resultsDict[chromoName] = {}
                    
                for anno in resultsDictByAnno:
                
                    if not(anno in resultsDict[chromoName]):
                        
                        acc, corr, tot = resultsDictByAnno[anno]
                        resultsDict[chromoName][anno] = [acc, corr, tot]
                    
            except IOError:
                print('No results found for this chromosome %s load file %s' %  (chromoName, loadFile))
     
    #dump the result (obs: the output dict has the same name as the one kept in 
    #each chromo's folder --- viz the files data were loaded right above and from 
    #which the dict we dump here was created)
    dumpFile = rootOutput +'accuracyByChromoAnnoDictionary'
    pickle.dump(resultsDict, open(dumpFile, "wb") )
        
            
    #Then aggr the acc's:
    returnDict = {}
    for chromoName in resultsDict:
        
        for anno in resultsDict[chromoName]:
        
            if not(anno in returnDict):

                returnDict[anno] = resultsDict[chromoName][anno]
            
            else:
                
                returnDict[anno][0] += resultsDict[chromoName][anno][0] #useless actually; these fig's are the acc's which cannot just be added; we calc the acc below
                returnDict[anno][1] += resultsDict[chromoName][anno][1] #the corr fig's
                returnDict[anno][2] += resultsDict[chromoName][anno][2] #the tot fig's
                
    #Compute the acc:
    for anno in returnDict:
        
        acc, corr, tot = returnDict[anno]
        #refresh the acc
        returnDict[anno][0] = float(corr)/tot
        
    #dump the result
    dumpFile = rootOutput +'accuracyByAnnoDictionary'
    pickle.dump(returnDict, open(dumpFile, "wb") )
        

def getInfo_calculateAggrAccOnChromos(rootOutput, chromosomeOrderList, chromosomeDict, annotationList, rootGenome, dictionaryNameAggr = 'accuracyByChromoAnnoDictionary', dictionaryNameSeg = 'accuracyByChromoAnnoSegDictionary', inclChromoLengthInTable_b = 1, captionText = 'Statistics on chromosome input', fileName = 'table_chromoStats.txt'):
    '''collect the info --- counts of nr of qualified for the various annotations --- on which
    the output of calculateAggrAccOnChromos is based.
    
    Obs: chromosomeDict is only supplied to provide the start/end positions used.'''
    
    #load the dict's:
    loadFile = rootGenome + 'chromosomeLengthDict'
    chromoLengthDict = pickle.load(open(loadFile, "rb"))
    print("chromoLengthDict ", chromoLengthDict)
    
    loadFile = rootOutput + dictionaryNameAggr
    resultsDict = pickle.load(open(loadFile, "rb"))
    print("accuracyByChromoAnnoDictionary keys ", resultsDict.keys())
    
    loadFile = rootOutput + dictionaryNameSeg
    resultsDictSeg = pickle.load(open(loadFile, "rb"))
    print("accuracyByChromoAnnoSegDictionary keys ", resultsDict.keys())
    
    outputDict = {}

    for chromo in chromosomeOrderList:
        
        startPos, endPos = chromosomeDict[chromo]
        
        chromoLength = chromoLengthDict[chromo]
        
        outputDict[chromo] = {}
        
        #get the total count of all qualified positions in this chromo:
        acc_all, corr_all, tot_all = resultsDict[chromo]['all']
        
        #to find the number of segments had:
        nrSegments = max(resultsDictSeg[chromo]['all'].keys()) + 1
        
        if inclChromoLengthInTable_b == 1:
            outputDict[chromo]['length'] = chromoLength
            
        outputDict[chromo]['#qualified'] = tot_all 
        outputDict[chromo]['fraction qualified'] = float(tot_all)/chromoLength
        outputDict[chromo]['#segments'] = nrSegments
        
        for anno in resultsDict[chromo]:
            
            acc, corr, tot = resultsDict[chromo][anno]
            
            #compute the desired fractions:
            outputDict[chromo][anno] = float(tot)/tot_all
            
            if annotationList.count(anno) < 1:
                print("Warning: %s is not in the annotationList. Is appended." % anno)
                annotationList.append(anno)
            
    #make tex-table:
    colNames = []   
    if inclChromoLengthInTable_b == 1:
        colNames = ['length']
    colNames.extend(['#qualified', 'fraction qualified', '#segments'])
    colNames.extend(annotationList)
    makeTexTable(inputDict = outputDict, rowColHeading = 'chr', rowNames = chromosomeOrderList, columnNames = colNames, captionText = captionText,  decPoints = 3, rootOutput = rootOutput, fileName = fileName)
    
    return outputDict
    

#Addendum to calculateAggrAccOnChromos: aggregates on a given partition of the chromosomes into two
#parts. To be used for maiking the odd/even chromos split.
def calculateAggrAccOnChromosPartition(rootOutput, chromosomeDict, chromoPartition, dictionaryName = 'accuracyChromoByAnnoDictionary'):
    '''Computes the acc accross all chromos in each of the two parts of the partition, for each annotation type covered by the dict
    accuracyByChromoAnnoDictionary generated with calculateAggrAccOnChromos.
    
    chromoPartition: a list of two lists, [part1, part2], where part1 and part2 is a partition of the chromosomes (covered in the accuracyByChromoAnnoDictionary dict).
    '''

    #first load the dictionary generated with calculateAggrAccOnChromos:
    loadFile = rootOutput +'accuracyByChromoAnnoDictionary'
    resultsDict = pickle.load(open(loadFile, "rb") )

    #Then aggr the acc's:
    part1, part2 = chromoPartition
    returnDict = {}
    returnDict['part1'] = {}
    returnDict['part2'] = {}
    for chromoName in resultsDict:
        
        if part1.count(chromoName) > 0:
        
            for anno in resultsDict[chromoName]:
            
                if not(anno in returnDict['part1']):
    
                    returnDict['part1'][anno] = resultsDict[chromoName][anno]
                
                else:
                    
                    returnDict['part1'][anno][0] += resultsDict[chromoName][anno][0] #useless actually; these fig's are the acc's which cannot just be added; we calc the acc below
                    returnDict['part1'][anno][1] += resultsDict[chromoName][anno][1] #the corr fig's
                    returnDict['part1'][anno][2] += resultsDict[chromoName][anno][2] #the tot fig's

        elif part2.count(chromoName) > 0:
        
            for anno in resultsDict[chromoName]:
            
                if not(anno in returnDict['part2']):
    
                    returnDict['part2'][anno] = resultsDict[chromoName][anno]
                
                else:
                    
                    returnDict['part2'][anno][0] += resultsDict[chromoName][anno][0] #useless actually; these fig's are the acc's which cannot just be added; we calc the acc below
                    returnDict['part2'][anno][1] += resultsDict[chromoName][anno][1] #the corr fig's
                    returnDict['part2'][anno][2] += resultsDict[chromoName][anno][2] #the tot fig's
        

    #Compute the acc:
    for part in returnDict:
        for anno in returnDict[part]:
            
            acc, corr, tot = returnDict[part][anno]
            #refresh the acc
            returnDict[part][anno][0] = float(corr)/tot
            
    #dump the result
    dumpFile = rootOutput +'accuracyByAnnoDictionaryInChromoPartition'
    pickle.dump((returnDict,chromoPartition), open(dumpFile, "wb") )
     
    return returnDict
    

def makePlotOfAggrAccOnChromosPartitionSeveralModels(rootPredictModelList, modelNameList, partitionNameList, rootOutput, offsetValue = 0.0, nameChangeDict = modelNameChangeDict ,saveAtDpi = 300, coloDict = colorDict, fontSizeLabels = 'medium',  legend_b = 1):
    '''  Makes bar plot of the accuracies in a partition of the chromosomes, as returned by
    calculateAggrAccOnChromosPartition for some set of models. 

    Assumes the partition consists of two parts.'''
    
    
    returnDict = {}
    
    #load the results for all models into a single dict, mapping each model to the list of acc's from the two partitions:
    resultsDict = {}
    modelCnt = 0
    for modelName in modelNameList:
        
        loadFile = rootPredictModelList[modelCnt] + 'accuracyByAnnoDictionaryInChromoPartition'
        partitionAccDict, chromoPartition = pickle.load(open(loadFile,"rb"))
        
        for part in partitionAccDict:
            
            if not(modelName in resultsDict):
                resultsDict[modelName] =  []
                
            resultsDict[modelName].append(partitionAccDict[part]['all'][0]  - offsetValue)
            
            if not(modelName in returnDict):
                returnDict[modelName] = {}
                
            returnDict[modelName][part] = partitionAccDict[part]['all'][0]
    
        modelCnt += 1

    print(resultsDict.keys())
    
    nModels = len(modelNameList)
            
    X = np.array([0.5, 1]) #we assume a partition in two parts
    bar_width = 0.5*1./(nModels +1)

    fig, ax = plt.subplots()
#        ax = fig.add_axes([0,0,1.2,1.2])
    
#        print(dataDict)
    
    cnt = 0
    for modelName in modelNameList:
        
        modelNameForLabel = nameChangeDict[modelName]
    
        ax.bar(X + cnt*bar_width , resultsDict[modelName], color = colorDict[modelName], width = bar_width, label = modelNameForLabel)
        
        #add tw horizontal lines indicating the acc levels of LSTM4S odd/even
        if modelName == 'LSTM4S':
                                
            plt.axhline(y=resultsDict[modelName][0], color='black', linestyle='--', linewidth = 0.5)
            
            plt.axhline(y=resultsDict[modelName][1] , color='black', linestyle='-.', linewidth = 0.5)

        
        cnt += 1
    
    #set the labels of the yticks right, corr to the offset:
    plt.yticks(fontsize = fontSizeLabels)
    plt.draw()
    yTicks = ax.get_yticks()
    yTickLabels = [item.get_text() for item in ax.get_yticklabels()]
    #print(yTickLabels)
    yTickLabelsNew = [str(float(yLbl) + offsetValue)  for yLbl in yTickLabels]
    
    plt.yticks(yTicks,yTickLabelsNew, fontsize = fontSizeLabels)    
    
    ax.set_ylabel('Accuracy', fontsize = fontSizeLabels)
#        ax.set_title('...')
#        ax.set_xticks(X + bar_width / 2)
    plt.xticks(X + (nModels-1)*0.5*bar_width,partitionNameList, rotation = 90, fontsize = fontSizeLabels)
    ax.set_xlabel('Partition', fontsize = fontSizeLabels)
    if legend_b == 1:
        ax.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = fontSizeLabels) #places legend outside the frame with the upper left placed at coords (1.05, 0.6)
    
    plt.tight_layout()
    #plt.show()
        
    plt.savefig(rootOutput + 'models_by_chromoPartition.pdf', dpi=saveAtDpi)
    
    plt.close()
    
    return returnDict
    
    
    

def collectAccuracyAnnotationsSeveralModels(rootOutput, 
                                            rootPredictModelList = [], 
                                  modelFileList = [],
                                  modelNameList = [],
                                    nameChangeDict = modelNameChangeDict,
                                  annotationOrderList = [],
                                    plot_b = 0,
                                    addAvg_b = 0,
                                    avgLevel = 0,
                                    saveAtDpi = 100):
    '''Collects the accuracy per chromo for each model in the input list (for which 
    the acc-dict as created by calculateAggrAccOnChromos are had). 
    So we just use, for each model, the accuracyByAnnoDictionary. Returns the info 
    in a dict mapping each chromo to model to accuracy.
    
    If plot_b = 1: generates and saves a bar-plot shownig the comparison.
    
    If addAvg_b = 1: a horizontal line at the level of the average performance of
    the first model in the modelFileList is added (broken black).
    '''

    returnDict = {}
    
    modelCnt = 0
    for model in modelFileList:
        
        loadFile = rootPredictModelList[modelCnt] + 'accuracyByAnnoDictionary'
        totAccDict = pickle.load(open(loadFile,"rb"))
        print(totAccDict)
        
        modelName = modelNameList[modelCnt]
        for anno in totAccDict:
            
            if not(anno in returnDict):
                returnDict[anno] =  {}
                
            returnDict[anno][modelName] = totAccDict[anno]
    
        modelCnt += 1

    print(returnDict.keys())
    
    #dump the result
    dumpFile = rootOutput +'accuracyAnnoByModelDictionary'
    pickle.dump(returnDict, open(dumpFile, "wb") )
    
    if plot_b == 1:
        
        if not(annotationOrderList):
            
            annoList = []
            for anno in returnDict:
                                    
                if annoList.count(anno) == 0:
                            
                    annoList.append(anno)
        else:
            
            if set(annotationOrderList) != set(returnDict.keys()):
                
                print("Warning: the provided rowNames differ from the keys of the input dict")
                raw_input("Press any key if you want to continue")
            
            annoList = annotationOrderList
        
        nAnnos = len(annoList)
        print("nAnnos ", nAnnos)
        nModels = modelCnt
        
        #"transpose" the dict:
        dataDict = {}
        for anno in annoList:
            
            for modelName in returnDict[anno]:
                
                if modelName in dataDict:
                    
                    dataDict[modelName].append(returnDict[anno][modelName][0])
                
                else:
                
                    dataDict[modelName] = [returnDict[anno][modelName][0]]
                
        
        #colors = cm.get_cmap('Set2')
        
        X = np.arange(nAnnos)
        bar_width = 1./(nModels +1)

        fig, ax = plt.subplots()
#        ax = fig.add_axes([0,0,1.2,1.2])
        
#        print(dataDict)
        
        cnt = 0
        for modelName in modelNameList:
            
            labelModelName = nameChangeDict[modelName]
        
            ax.bar(X + cnt*bar_width , dataDict[modelName], color = colorDict[modelName], width = bar_width, label = labelModelName)
            
            if addAvg_b == 1 and cnt == 0: 
                                
                plt.axhline(y=avgLevel, color='black', linestyle='--', linewidth = 0.5)
            
            cnt += 1
        
        plt.yticks(fontsize = 'x-small')
        ax.set_ylabel('Accuracy', fontsize = 'x-small')
#        ax.set_title('...')
#        ax.set_xticks(X + bar_width / 2)
        plt.xticks(X + (nModels-1)*0.5*bar_width,annoList, rotation = 90, fontsize = 'x-small')
        ax.set_xlabel('Annotation', fontsize = 'x-small')
        ax.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = 'x-small') #places legend outside the frame with the upper left placed at coords (1.05, 0.6)
        
        plt.tight_layout()
        plt.show()
            
        plt.savefig(rootOutput + 'models_by_anno.pdf', dpi=saveAtDpi)

    return returnDict



def collectAccuracyChromosomesSeveralModels(rootOutput, 
                                            rootPredictModelList = [], 
                                  modelFileList = [],
                                  modelNameList = [],
                                    nameChangeDict = modelNameChangeDict,
                                  chromosomeOrderList = [],
                                    plot_b = 0, 
                                    addAvg_b = 0,
                                    avgLevel = 0,
                                    saveAtDpi = 100):
    '''Collects the accuracy per annotation for each model in the input list (for which 
    the acc-dict as created by getAccuracyChromosomes/getAccuracyOnSegments are had). 
    So we just take out, for each model, the sub-dict of accuracyByChromoAnnoDictionary for 
    annotation = 'all'. Returns the info in a dict mapping each chromo to model to accuracy.
    
    If plot_b = 1: generates and saves a bar-plot shownig the comparison.
    
    If addAvg_b = 1: a horizontal line at the level of the average performance of
    the first model in the modelFileList is added (broken black).
    '''

    returnDict = {}
    
    modelCnt = 0
    for model in modelFileList:
        
        loadFile = rootPredictModelList[modelCnt] + 'accuracyByChromoAnnoDictionary'
        totAccDict = pickle.load(open(loadFile,"rb"))
        print(totAccDict)
        
        modelName = modelNameList[modelCnt]
        for chromo in totAccDict:
            
            #hack to cover the case of LSTM4P, which isn't  predicted on the full chromos but e.g. on hg38_part2_chr1:
            origChromo = chromo 
            if modelName == 'LSTM4P':
                chromo = chromo[:5] + chromo[11:]
            
            if not(chromo in returnDict):
                returnDict[chromo] =  {}
                
            returnDict[chromo][modelName] = totAccDict[origChromo]['all']
    
        modelCnt += 1
    
    #dump the result
    dumpFile = rootOutput +'accuracyChromoByModelDictionary'
    pickle.dump(returnDict, open(dumpFile, "wb") )
    
    if plot_b == 1:
        
        if not(chromosomeOrderList):
            
            chromoList = []
            for chromo in returnDict:
                                    
                if chromoList.count(chromo) == 0:
                            
                            chromoList.append(chromo)
        else:
            
            if set(chromosomeOrderList) != set(returnDict.keys()):
                
                print("Warning: the provided rowNames differ from the keys of the input dict")
                raw_input("Press anything if you want to continue")
            
            chromoList = chromosomeOrderList
        
        nChromos = len(chromoList)
        print("nChromos ", nChromos)
        nModels = modelCnt
        
        #"transpose" the dict:
        dataDict = {}
        for chromo in chromoList:
            
            for modelName in returnDict[chromo]:
                
                if modelName in dataDict:
                    
                    dataDict[modelName].append(returnDict[chromo][modelName][0])
                
                else:
                
                    dataDict[modelName] = [returnDict[chromo][modelName][0]]
                
        
#        colors = cm.get_cmap('Set2')
        
        X = np.arange(nChromos)
        bar_width = 1./(nModels +1)

        fig, ax = plt.subplots()
#        ax = fig.add_axes([0,0,1.2,1.2])
        
#        print(dataDict)
        
        cnt = 0
        for modelName in modelNameList:
            
            labelModelName = nameChangeDict[modelName]
        
            ax.bar(X + cnt*bar_width , dataDict[modelName], color = colorDict[modelName], width = bar_width, label = labelModelName)
            
            if addAvg_b == 1 and cnt == 0: 
                                
                plt.axhline(y=avgLevel, color='black', linestyle='--', linewidth = 0.5)

            cnt += 1
        
        plt.yticks(fontsize = 'x-small')
        ax.set_ylabel('Accuracy', fontsize = 'x-small')
#        ax.set_title('...')
#        ax.set_xticks(X + bar_width / 2)
        plt.xticks(X + (nModels-1)*0.5*bar_width,chromoList, rotation = 90, fontsize = 'x-small')
        ax.set_xlabel('Chromosome', fontsize = 'x-small')
        ax.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = 'x-small') #places legend outside the frame with the upper left placed at coords (1.05, 0.6)
        
        plt.tight_layout()
        plt.show()
            
        plt.savefig(rootOutput + 'models_by_chromo.pdf', dpi = saveAtDpi)

    return returnDict



def scatterplotAnnoByChromo(inputDict, chromosomeOrderList = [], nameChangeDict = {}, modelName = '', rootOutput = '', saveAtDpi = 100):
    '''inputDict: should map chromo to anno to value.'''

    if not(chromosomeOrderList):
        chromosomeOrderList = inputDict.keys()
        
    #get the possible annotations:
    annoList = []
    for chromo in inputDict:
        
        for anno in inputDict[chromo]:   
            
            if annoList.count(anno) == 0:
                
                annoList.append(anno)

        
    colors = cm.get_cmap('Set3')
    fig = plt.figure()
    
    cnt = 0
    for anno in annoList: 
    
        yValues = [inputDict[chromo][anno][0] for chromo in chromosomeOrderList]
        print(chromosomeOrderList)
        print(yValues)

        if anno in nameChangeDict:
            plt.scatter(chromosomeOrderList, yValues, c = colors(cnt+2), label = nameChangeDict[anno])
        else:
            plt.scatter(chromosomeOrderList, yValues, c = colors(cnt+2), label = anno)
        
        cnt += 1
    
    plt.xticks(rotation = 90, fontsize = 'x-small')
    plt.yticks(fontsize = 'x-small')
    plt.ylabel('Accuracy', fontsize = 'x-small')
    plt.xlabel('Chromosome', fontsize = 'x-small')    
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = 'x-small') #places legend outside the frame with the upper left placed at coords (1.05, 0.6)

    plt.tight_layout()
    plt.show()
    
    plt.savefig(rootOutput + modelName + '_scatterAnnoChromo.pdf', dpi = saveAtDpi)
    

def makeTexTable(inputDict, rowColHeading, captionText, columnNames = [], rowNames = [],  nameChangeDict = {},  inputDict2 = {}, decPoints = 4, rootOutput = '', fileName = ''):


    #If columnNames are not provided: Find the "union" of the "column" keys over 
    #all "rowKeys". Note that the order of the columns is defined either by the
    #columnNames list or by the list generated here:
    if not(columnNames):
        colKeys = []
        for rowKey in inputDict:
            
                for colKey in inputDict[rowKey]:
                    
                    if colKeys.count(colKey) == 0:
                        
                        colKeys.append(colKey)
    else:
        
        colKeys = columnNames
        
    #Similarly to the colKey/columnNames we generate a rowKeys list, which
    #gives also the order of appearance:
    if not(rowNames):
        rowKeys = []
        for rowKey in inputDict:
                                
            if rowKeys.count(rowKey) == 0:
                        
                        rowKeys.append(rowKey)
    else:
        
        if set(rowNames) != set(inputDict.keys()):
            
            print("Warning: the provided rowNames differ from the keys of the input dict")
            raw_input("Press anything if you want to continue")
        rowKeys = rowNames
    
        
        
    s = r'\begin{table}[h!]' + '\n'
    s +=r'  \begin{center}' + '\n'
    s +=r'    \label{tab:table1c}' + '\n'
    s +=r'    \begin{tabular}{l | c | c | r} % <-- Alignments: 1st column left, 2nd middle and 3rd right, with vertical lines in between' + '\n'
    
    subS =r'      \textbf{' + rowColHeading + '}'
    for colKey in colKeys:
        
        colName = colKey
        #replace if wanted:
        if colKey in nameChangeDict:
            colName = nameChangeDict[colKey]
        
        subS += r'& \textbf{' + colName + '}'
        
    subS += r'\\'   + '\n'
    
    s += subS    
    
    s +=r'      \hline' + '\n'
    
    for rowKey in rowKeys:
        
        subS = rowKey
        #replace if wanted:
        if rowKey in nameChangeDict:
            subS = nameChangeDict[rowKey]
  
    
        for colKey in colKeys:
            
            if not(colKey in inputDict[rowKey]):
                
                subS += r' & na'

            else:                
                
                try:
                    if isinstance(inputDict[rowKey][colKey][0], int):
                        subS += r' & ' + str(inputDict[rowKey][colKey][0])
                    else:
                        subS += r' & ' + str(round(inputDict[rowKey][colKey][0], decPoints))
                except TypeError:
                    if isinstance(inputDict[rowKey][colKey], int):
                        subS += r' & ' + str(inputDict[rowKey][colKey])
                    else:
                        subS += r' & ' + str(round(inputDict[rowKey][colKey], decPoints))
                except IndexError:
                    if isinstance(inputDict[rowKey][colKey], int):
                        subS += r' & ' + str(inputDict[rowKey][colKey])
                    else:
                        subS += r' & ' + str(round(inputDict[rowKey][colKey], decPoints))
                    
        subS += r'\\'   + '\n'
        
        s += subS        
        
    if inputDict2:
        
        subS = 'All'

        for colKey in colKeys:
            
            try:
                if isinstance(inputDict2[colKey][0], int):
                    subS += r' & ' + str(inputDict2[colKey][0])
                else:
                    subS += r' & ' + str(round(inputDict2[colKey][0], decPoints))
            except TypeError:
                if isinstance(inputDict2[colKey], int):
                    subS += r' & ' + str(inputDict2[colKey])
                else:
                    subS += r' & ' + str(round(inputDict2[colKey], decPoints))
            except IndexError:
                if isinstance(inputDict2[colKey], int):
                    subS += r' & ' + str(inputDict2[colKey])
                else:
                    subS += r' & ' + str(round(inputDict2[colKey], decPoints))
    
        subS += r'\\'   + '\n'
    
        s += subS        
             
    s +=r'    \end{tabular}' + '\n'
    s +=r'        \caption{'+ captionText + '}'  + '\n'
    s +=r'  \end{center}' + '\n'
    s +=r' \end{table}'
    
    
    outputFileName = rootOutput + fileName
    outputFile = open(outputFileName, 'w')
    outputFile.write(s)
    outputFile.close()
    
    return s, colKeys                                              
        

#util for randomizing the input arrays in the Fourier transfs:
def displaceIntervalsInSegments(inputArray, intervalsDict, segmentId, chromoName):
    '''
    inputArray: an array of floats/ints, typically a segment (segmentId) of prediction-prob's
    intervalsDict: as output by rndDisplaceAnnoIntervalsInSegments; must be obtained
    on the same chromosome as the inputArray.
    '''
#    ''' Obs: this function returns a changed version of the inputArray; the code could be
#    written using copying of the inputArray but in the present version this is not 
#    done (to avoid a possible memory-footprint from copying). '''
    
    inputArrayCopy = inputArray.copy()
        
    for idx in intervalsDict[chromoName][segmentId]:
        
        (start,end), (newStart, newEnd) = intervalsDict[chromoName][segmentId][idx]
#        print(segmentId, idx, start,end, newStart, newEnd)
        
        try:
            #rearrange the inputArray: 
            if end <= newStart or newEnd <= start:
                
#                print("inside 0")
#                print(inputArray[newStart:newEnd])
                
                tmp = inputArrayCopy[newStart:newEnd].copy()
                inputArrayCopy[newStart:newEnd] = inputArrayCopy[start:end]
                inputArrayCopy[start:end] = tmp
                
#                print(inputArrayCopy[newStart:newEnd])
            
            else: #newStart < end and start < newEnd 
                 #move the pieces around, 'cyclically'
                 if start <= newStart:
#                     print("inside1")
                     tmp = inputArrayCopy[end:newEnd].copy() #existing end
                     inputArrayCopy[newStart:newEnd] = inputArrayCopy[start:end]
                     inputArrayCopy[start:newStart] = tmp #existing end is moved up front
                 else:
#                     print("inside2")
                     tmp = inputArrayCopy[newStart:start].copy() #existing start
                     inputArrayCopy[newStart:newEnd] = inputArrayCopy[start:end]
                     inputArrayCopy[newEnd:end] = tmp #existing start is moved to the end
        except ValueError:
            print(segmentId, idx, start,end, newStart, newEnd)
            raw_input("Den prikkede ...")       
    
#    print(np.sum(np.abs(inputArrayCopy - inputArray)))
#    raw_input("Den storprikkede ...") 
    
    return inputArrayCopy
        
        



def computeFourierOnSegments(rootInput, 
                             rootOutput,
                             modelFileName, 
                             segmentLength,
                             genomeIdName, #for saving the prediction array; could eg be a chromo name
                             averageRevComplementary_b,
                             inputArrayType = 0,
                             ratioQcutoff = 0.9,
                             windowLength = 1,
                             stepSize = 1,
                             fourierStart = 1, 
                             fourierStop = 15000,
                             fourierWindowLength = 500,
                             fourierStep = 100,
                             fourierRawPlotFrq = 10,
                             plotOnlyNorm_b = 1,
                             getInvPeak_b = 0,
                             nPeak = 4500, #optional; for getting inv transform for coeff's in peak 
                             shuffle_b = 0,
                             randomizeDisqualified_b = 0,
                             randomizePositions_b = 0, #randomize a window around each position specified in positionsDict 
                             randomizingByShuffle_b = 0,
                             randomizingMultiplicative_b = 0,
                             randomizationScale = 1., 
                             positionsDict = {},
                             randomizeAnnoIntervals_b = 0,
                             randomizeAnnoIntervalsName = '',
                             randomizeAnnoIntervalsDict = {},
                             randomizePositionsWindow = 5000,
                             randomizePositionsName = 'randTSS',
                             forATorGCbias_b = 0,
                             rootInput_forATorGCbias = '',
                             rootOutput_forATorGCbias = '',
                             modelFileName_forATorGCbias = '',
                             fullGC_b = 0,
                             dumpFourier_b = 0,
                             dumpFileNamePrefix = '', #if dumping the Fourier coeffs for some modified version of the input arrays, set this prefix to reveal the content/modification
                             plots_b = 1, #if 0 no plots will be made and saved
                             fontSizeLabels = 'medium',
                             saveAtDpi = 100):
    '''Reads in average predictions (over windows of set length) previosly made onn segments of a
    genomic string and performs a Fourier transformation on each. 

    Input:    
    fourierStop: last frequency to show
    fourierWindowLength, fourierStep: take norm of Fourier coeff'c over a window of size fourierWindowLength in steps of fourierStep
    inputArrayType: integer determining which type of arrays is used as input: if 0 the
                    input arrays are taken to be the predReturns; if 1 the reference base
                    probabilities are used (ie the predArrays dotted by the labelArrays)
    
    randomizingMultiplicative_b: if 1 the input arrays will be randomized by multiplying with a random float in [0,1], else by replacing by such a value
    
    randomizeAnnoIntervals_b: only works for a single chromosme at a time. 
    randomizeAnnoIntervalsDict: as output by rndDisplaceAnnoIntervalsInSegments (single chromo)
    
    Output: two plots
    1) shows .. 
    2) show norm of coeff's in windows (size fourierWindowLength) stepping fourierStep, from 0 to frequency fourierStop
    '''

    outputList = []    
    
    data_b = 0
    i = 0
    
    if shuffle_b == 1:
        raw_input("I'll shuffle the input arrays!")
        
    if randomizeDisqualified_b == 1:
        raw_input("I'll randomize the disqualified part of the input arrays!")
        if randomizingByShuffle_b == 1:
            raw_input("... which I'll do by shuffling/random replacement")
        else:
            raw_input("... which I'll do by random scalar [0.25 ,0.35] replacement")

        
    if randomizePositions_b == 1:
        raw_input("I'll randomize windows around the provided positions of the input arrays!")
        if randomizingByShuffle_b == 1:
            raw_input("... which I'll do by shuffling/random replacement")
        else:
            raw_input("... which I'll do by random scalar [0,1] replacement")
            
        
    if randomizingMultiplicative_b == 1:
        randomizePositionsName  = randomizePositionsName + '_multiplicative'  
        
    if randomizeAnnoIntervals_b == 1:
        raw_input("I'll displace the input arrays randomly acc to the provided randomizeAnnoIntervalsDict!")

    
    while data_b == 0:
        
        genomeIdNameSeg = genomeIdName + '_seg' + str(int(segmentLength)) + '_segment' + str(i)
        
        if forATorGCbias_b == 0:
            
            if inputArrayType == 0:
                loadFile = rootInput +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   

            elif inputArrayType == 1:
                loadFile = rootInput +  modelFileName + '_predArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
                
        else:
            
            loadFile = rootInput_forATorGCbias +  modelFileName_forATorGCbias + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
        
#        loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)        
        
        print(loadFile)
    #    storedPredThisSeg = pickle.load(open(loadFile,"rb"))
    #    if storedPredThisSeg:
    #        avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg
    #        print(avgPredSeg.shape)
    #        
    #        #get the qualified array too:
    #        loadFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
    #        qualArraySeg = pickle.load(open( loadFile, "rb" ) )
    #        print(qualArraySeg.size)
    #
    #
    #        data_b = 1
            
        try:
                        
            if inputArrayType == 0:
                
                storedPredThisSeg = pickle.load(open(loadFile,"rb"))
                
                avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg
            
        
    #            avgPredSeg = storedPredThisSeg 
                print(i, avgPredSeg.shape)
    #            print(avgPredSeg[100000:101000])
    #            raw_input("S er den dejligste")
                
                #we need the qualified array:
                loadFileQual = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                qualArraySeg = pickle.load(open( loadFileQual, "rb" ) )
                print(qualArraySeg.size)
                
                L = avgPredSeg.shape[0] #for later use -- eg if randomizing by shuffling
                                
            elif inputArrayType == 1: #in this case we want the array to contain the probability of the ref base assigned by the model
                
                #the prob of the ref base is obtained simply by taking the dot-product of 
                #the predArray and the label array, which we therefore also load:
                predArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                loadFile = rootInput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                labelArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                #we need the qualified array:
                loadFileQual = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                qualArraySeg = pickle.load(open( loadFileQual, "rb" ) )
                print(qualArraySeg.size)                
                
                
                #Modify the label array at positions having a non-ACGT letter; as it is, the label
                #is set to (3,3,3,3), but we want to let it sum to 1 and represent an unknown letter,
                #so we set it to (1/4, 1/4, 1/4, 1/4) there. The desired positions are among the 
                #the disqualified so we only need to scan those:
                disqIdxs = np.where(qualArraySeg == 0)[0]
#                print("disq idxs ", disqIdxs)
                for dIdx in disqIdxs:

                    if np.sum(labelArrayThisSeg[dIdx]) > 2:
                        labelArrayThisSeg[dIdx] = np.asarray([0.25, 0.25, 0.25, 0.25])

#                for l in range(L):
#                    
#                    if np.sum(labelArrayThisSeg[l]) > 2:
#                        labelArrayThisSeg[l] = np.asarray([0.25, 0.25, 0.25, 0.25])
                        
                #take the entry-wise dot product; we use avgPredSeg as name for convenience (since used for predReturn/inputArrayType=0-case ) 
                L = labelArrayThisSeg.shape[0]
                avgPredSeg = np.zeros(shape = L, dtype = 'float64')
                for l in range(L):
                    avgPredSeg[l] = np.dot(predArrayThisSeg[l],labelArrayThisSeg[l])
           
#            #get the qualified array too:
#            loadFile = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
#            qualArraySeg = pickle.load(open( loadFile, "rb" ) )
#            print(qualArraySeg.size)
            
            if fullGC_b == 1 and forATorGCbias_b == 1:
                    
                loadFile = rootInput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                labelArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                avgPredSeg = dataUtils.getFullGCcontentArray(GCcontentArray = avgPredSeg, labelArray = labelArrayThisSeg , qualArray = qualArraySeg)
            
            #shuffle the two arrays if desired:
            if shuffle_b == 1:
                
                shuffledIdxs = np.arange(avgPredSeg.shape[0])
                np.random.shuffle(shuffledIdxs)
                
                avgPredSeg = avgPredSeg.take(shuffledIdxs)
                qualArraySeg = qualArraySeg.take(shuffledIdxs)
                
            #randomize the content at all non-qualified positions:
            if randomizeDisqualified_b == 1:

                #indices of non-qual's:
                disQidxs = np.where(qualArraySeg == 0)[0]
                print("disQidxs ", disQidxs)
                
                print("Before randomizing ", avgPredSeg.take(disQidxs) )
#                if len(disQidxs) > 1: print("Gylle ", avgPredSeg[disQidxs[0]:(disQidxs[1] + 1)])
                if randomizingByShuffle_b == 1:
                    for idx in disQidxs:
                        idxRnd = np.random.randint(0, L, 1)[0] #pick a random index not above the length of the avgPredSeg -1 
                        avgPredSeg[idx] = avgPredSeg[idxRnd]
                else:
                    for idx in disQidxs:
                        
                        avgPredSeg[idx] = np.random.random(1)*0.1 + 0.25 #random flot in [0,1), which is fine
                
                print("After randomizing ", avgPredSeg.take(disQidxs))
                
            if randomizePositions_b == 1: 
                
                #get indices for this chromo and segment:
                idxs = positionsDict[genomeIdName][i]
                print("I'll randomize at ", idxs)
                
                print("before randomizing ", avgPredSeg.take(idxs))
                #randomize:
                for idx in idxs:

                    intervalStart = max(idx - randomizePositionsWindow, 0)
                    intervalEnd = min(idx + randomizePositionsWindow, segmentLength)                    
                    try:        
                        if randomizingByShuffle_b == 1:
                            for j in range(intervalStart, intervalEnd+1):
                                jRnd = np.random.randint(0, L, 1)[0] #pick a random index not above the length of the avgPredSeg -1 
                                avgPredSeg[j] = avgPredSeg[jRnd]
                        elif randomizingMultiplicative_b == 1:
                            for j in range(intervalStart, intervalEnd+1):
                                avgPredSeg[j] = np.random.random(1)*avgPredSeg[j]*randomizationScale
                        else:
                            for j in range(intervalStart, intervalEnd+1):
                                avgPredSeg[j] = np.random.random(1)*randomizationScale
                    except IndexError:
                        print("When randomizing postions: Index error at idx", idx)
                        continue
                print("after randomizing ", avgPredSeg.take(idxs))
                
            
            if randomizeAnnoIntervals_b == 1:
                
                 avgPredSeg = displaceIntervalsInSegments(inputArray = avgPredSeg.copy(), intervalsDict = randomizeAnnoIntervalsDict, chromoName = genomeIdName, segmentId = i )
                                         
            data_b = 1
                
        except IOError:
            i += 1
            if i > 10: 
                print("Files not found, last tried: ", loadFile)
                return
        except EOFError:
            i += 1
            if i > 1000:
                print("Empty file, last tried: ", loadFile)
                return
            continue
            
            

    if plots_b == 1:
        fig0, ax0 = plt.subplots()
        fig1_loc, ax1_loc = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2_loc, ax2_loc = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
    
    L = avgPredSeg.size
    print("avgPredArray size: ", L)
    segStart = 0
    cnt = 0
    while data_b == 1:
        
        if (cnt+1)%fourierRawPlotFrq == 0 and plots_b == 1:
            
            #save "local" plots so far (ie only for a bundle of segments) and reset:  
            if forATorGCbias_b == 0:

                if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput + 'Fourier_raw_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    


#                    fig4.savefig(rootOutput + 'InvFourier_peak_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')                
                
                elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput + 'Fourier_raw_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                    
#                    fig4.savefig(rootOutput + 'InvFourier_peak_shuffled_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')
            
                elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput + 'Fourier_raw_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput + 'Fourier_raw_end_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                    
                    
                elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput + 'Fourier_raw_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput + 'Fourier_raw_end_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
               

                elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput + 'Fourier_raw_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput + 'Fourier_raw_end_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    


               
            
            elif forATorGCbias_b == 1:

                if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                                    
#                    fig4.savefig(rootOutput_forATorGCbias + 'InvFourier_peak_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')


                elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                    
                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

#                    fig4.savefig(rootOutput_forATorGCbias + 'InvFourier_peak_shuffled_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')
    
                elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:

                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                    

                elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:

                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    


                elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:

                    if plotOnlyNorm_b == 0:
                        fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                    
            #reset
            segStart = cnt + 1 
            fig0, ax0 = plt.subplots()
            fig1_loc, ax1_loc = plt.subplots()
            fig2_loc, ax2_loc = plt.subplots()
            fig3, ax3 = plt.subplots()
            #fig4, ax4 = plt.subplots()
                    
            
        
        ratioQ = float(np.sum(qualArraySeg))/qualArraySeg.size
        
        print("ratioQ: ", ratioQ)
        
        if ratioQ > 0.9999:
            lineStyle = 'solid'
        else:
            lineStyle = 'dashed'
            
        ratioQlabel = str(round(ratioQ,4))
        
        if ratioQ > ratioQcutoff:
            
            #Fourier transform it:
            #scipy fast Fourier transform; input is real so we really need only the rfft (no 
            #need to do the fft); however, the output of the rfft is less straightforward than 
            #that of the fft, and it's not so heavy to run the fft here anyhow.
            fftAvgPred = fft(avgPredSeg) 
            N =  fftAvgPred.size
            print("Fourier transf lenght: ", N)
            
            if dumpFourier_b == 1:
                
                if forATorGCbias_b == 0:
                    dumpFile = rootOutput + dumpFileNamePrefix + 'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
                    pickle.dump(fftAvgPred, open(dumpFile, "wb"))
                
                elif forATorGCbias_b == 1:
                    dumpFile = rootOutput_forATorGCbias + dumpFileNamePrefix +  'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
                    pickle.dump(fftAvgPred, open(dumpFile, "wb"))
            
            frqs = np.fft.fftfreq(fftAvgPred.size)
            print("frqs ", frqs  )

            stop = fourierStop
            start = fourierStart

            print("First 10 (positive) frqs: " ,frqs[:10])
            print("Largest (positive) frq: %f" % frqs[int(float(N)/2)])
            print("First 10 (negative) frqs: " ,frqs[(N-10):N][::-1])
            print("Up to stop (%d) with frq %f, the max (min) modulus of Fourier coeff is: %f (%f)" % (stop, frqs[stop], np.max(np.abs(fftAvgPred[:stop])), np.min(np.abs(fftAvgPred[:stop]))) )
            print("After stop (%d) the max (min) modulus of Fourier coeff is: %f (%f)" % (stop, np.max(np.abs(fftAvgPred[stop:(int(float(N)/2))])), np.min(np.abs(fftAvgPred[stop:(int(float(N)/2))]))) )
        

            if plots_b == 1:
                
                Xs = np.arange(start = start, stop = stop, step = fourierStep)
                YAbs = [np.abs(fftAvgPred[n]) for n in Xs]
                yMaxs = [np.max(np.abs(fftAvgPred[n:stop])) for n in Xs]
        
                yNorm = [np.linalg.norm(fftAvgPred[n:(n+fourierWindowLength)]) for n in Xs]
        
                #fourier coeff's plot:
                p_ax0 = ax0.plot(Xs, YAbs, label = i)
                ax0.legend(loc = 'upper right', fontsize = fontSizeLabels)
                ax0.set_xlabel('frequency', fontsize = fontSizeLabels)
                ax0.set_ylabel('abs amplitude', fontsize = fontSizeLabels)
                colorForThisSegment = p_ax0[0].get_color()
                
                #max coeff to-end plot:
                ax1_loc.plot(Xs, yMaxs, label = i, color = colorForThisSegment)
                ax1_loc.set_xlabel('frequency', fontsize = fontSizeLabels)
                ax1_loc.set_ylabel('max abs mplitude', fontsize = fontSizeLabels)
                ax1_loc.legend(loc = 'upper right', fontsize = fontSizeLabels)
                ax1.plot(Xs, yMaxs, color = colorForThisSegment)
                ax1.set_xlabel('frequency', fontsize = fontSizeLabels)
                ax1.set_ylabel('max abs mplitude', fontsize = fontSizeLabels)
                
                if plotOnlyNorm_b == 0:
                    if forATorGCbias_b == 0:
                        
                        if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput + 'Fourier_maxToEnd_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                            fig1.savefig(rootOutput + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    
                    elif forATorGCbias_b == 1:
                        
                        if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                            fig1.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                        elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                            fig1.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
    
                    
                #local coeff's norm plot
                ax2_loc.plot(Xs, yNorm, linestyle = lineStyle, alpha = ratioQ, label = str(i) + ' , Q: ' + ratioQlabel, color = colorForThisSegment)
                ax2_loc.set_xlabel('frequency', fontsize = fontSizeLabels)
                ax2_loc.set_ylabel('amplitude (norm)', fontsize = fontSizeLabels)
                ax2_loc.legend(loc = 'upper right', fontsize = fontSizeLabels)
                ax2.plot(Xs, yNorm, linestyle = lineStyle, alpha = ratioQ, color = colorForThisSegment)
                ax2.set_ylabel('amplitude (norm)', fontsize = fontSizeLabels)
                ax2.set_xlabel('frequency', fontsize = fontSizeLabels)
                #ax2.legend(loc = 'upper right', fontsize = 'small')
                
                if forATorGCbias_b == 0:
                    
                    if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                        fig2.savefig(rootOutput + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
    
    
                elif forATorGCbias_b == 1:
                    
                    if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                        fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                        fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)
                          
                        
                    
                #Plot the end of spectrum too; since the input is real the fft-output is symmetric (only half is needed)
                #The tail of the spectrum ends at N/2 (we show the last 20pct):
    #            XsEnd = np.arange(start = stop, stop = N/2, step = fourierStep)
                XsEnd = np.arange(start = stop, stop = N, step = fourierStep)
    #            Ys = [fftAvgPred[n] for n in Xs]
                YAbsEnd = [np.abs(fftAvgPred[n]) for n in XsEnd]
                ax3.plot(XsEnd, YAbsEnd, label = i, color = colorForThisSegment)
                ax3.set_xlabel('frequency', fontsize = fontSizeLabels)
                ax3.set_ylabel('amplitude (norm)', fontsize = fontSizeLabels)
                ax3.legend(loc = 'upper right', fontsize = fontSizeLabels)
            
            
            #get part of inverse Fourier transf corr to one of the peaks. We take the coeff's for 
            #frq's in the peak and their negative frq counterparts (the inverse is then a real 
            #valued fct):
            if getInvPeak_b == 1:
                x = np.zeros(shape = qualArraySeg.shape)
                x[nPeak:(nPeak+fourierWindowLength)] = fftAvgPred[nPeak:(nPeak+fourierWindowLength)]
                x[-(nPeak+fourierWindowLength -1 ):(-nPeak + 1)] = fftAvgPred[-(nPeak+fourierWindowLength -1):(-nPeak +1)]
                print(x[nPeak:(nPeak+10)], x[-(nPeak+10 -1):(-nPeak +1)] )
                g = ifft(x)
                #dump it:
                if forATorGCbias_b == 0:
                    dumpFile = rootOutput + 'InvFourierTransform_Peak_' + str(nPeak) + '_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
                elif forATorGCbias_b == 1:
                    dumpFile = rootOutput_forATorGCbias + 'InvFourierTransform_Peak_' + str(nPeak) + '_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment'  + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
                
                pickle.dump(g, open(dumpFile, "wb"))
                
                
                
        i += 1
        genomeIdNameSeg = genomeIdName + '_seg' + str(int(segmentLength)) + '_segment' + str(i)

        if forATorGCbias_b == 0:
            
            if inputArrayType == 0:
                loadFile = rootInput +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
            
            elif inputArrayType == 1:
                loadFile = rootInput +  modelFileName + '_predArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)  
                
        else:
            
            loadFile = rootInput_forATorGCbias +  modelFileName_forATorGCbias + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   

        
        print(loadFile)
        data_b = 0 #not necc, really, since the code hits a return, in the exception to this "try" (ie when no loadfile is found) 
        try:
                        
            if inputArrayType == 0:
                
                storedPredThisSeg = pickle.load(open(loadFile,"rb"))
    
                avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg
    #            avgPredSeg = storedPredThisSeg
                print(i, avgPredSeg.shape)
    #            print(avgPredSeg[100000:101000])
    #            raw_input("S er den dejligste")
                
                #we need the qualified array:
                loadFileQual = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                qualArraySeg = pickle.load(open( loadFileQual, "rb" ) )
                print(qualArraySeg.size)
                
                L = avgPredSeg.shape[0] #for later use -- eg if randomizing by shuffling
                
            
            elif inputArrayType == 1: #in this case we want the array to contain the probability of the ref base assigned by the model
                
                #the prob of the ref base is obtained simply by taking the dot-product of 
                #the predArray and the label array, which we therefore also load:
                predArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                loadFile = rootInput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                labelArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                #we need the qualified array:
                loadFileQual = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                qualArraySeg = pickle.load(open( loadFileQual, "rb" ) )
                print(qualArraySeg.size)
                
                #Modify the label array at positions having a non-ACGT letter; as it is, the label
                #is set to (3,3,3,3), but we want to let it sum to 1 and represent an unknown letter,
                #so we set it to (1/4, 1/4, 1/4, 1/4) there. The desired positions are among the 
                #the disqualified so we only need to scan those:
                disqIdxs = np.where(qualArraySeg == 0)[0]
                for dIdx in disqIdxs:
                    
                    if np.sum(labelArrayThisSeg[dIdx]) > 2:
                        labelArrayThisSeg[dIdx] = np.asarray([0.25, 0.25, 0.25, 0.25])
#                L = labelArrayThisSeg.shape[0]
#                for l in range(L):
#                    
#                    if np.sum(labelArrayThisSeg[l]) > 2:
#                        labelArrayThisSeg[l] = np.asarray([0.25, 0.25, 0.25, 0.25])
                    
                
                #take the entry-wise dot product; we use avgPredSeg as name for convenience (since use for predReturn/inputArrayType=0-case ) 
                L = labelArrayThisSeg.shape[0]
                avgPredSeg = np.zeros(shape = L, dtype = 'float64')
                for l in range(L):
                    avgPredSeg[l] = np.dot(predArrayThisSeg[l],labelArrayThisSeg[l])
 
            
#            #get the qualified array too:
#            loadFile = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
#            qualArraySeg = pickle.load(open( loadFile, "rb" ) )
#            print(qualArraySeg.size)
            
            if fullGC_b == 1 and forATorGCbias_b == 1:
                    
                loadFile = rootInput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                labelArrayThisSeg = pickle.load(open(loadFile,"rb"))
                
                avgPredSeg = dataUtils.getFullGCcontentArray(GCcontentArray = avgPredSeg, labelArray = labelArrayThisSeg , qualArray = qualArraySeg)
            
            
            #shuffle the two arrays if desired:
            if shuffle_b == 1:
                
                shuffledIdxs = np.arange(avgPredSeg.shape[0])
                np.random.shuffle(shuffledIdxs)
                
                avgPredSeg = avgPredSeg.take(shuffledIdxs)
                qualArraySeg = qualArraySeg.take(shuffledIdxs)
                
            #randomize the content at all non-qualified positions:
            if randomizeDisqualified_b == 1:

                #indices of non-qual's:
                disQidxs = np.where(qualArraySeg == 0)[0]
                
                print("Before randomizing ", avgPredSeg.take(disQidxs) )
                if len(disQidxs) > 1: print("Gylle ", avgPredSeg[disQidxs[0]:(disQidxs[1] + 1)])
                if randomizingByShuffle_b == 1:
                    for idx in disQidxs:
                        idxRnd = np.random.randint(0, L, 1)[0] #pick a random index not above the length of the avgPredSeg -1 
                        avgPredSeg[idx] = avgPredSeg[idxRnd]
                else:
                    for idx in disQidxs:
                        
                        avgPredSeg[idx] = np.random.random(1)*0.1 + 0.25 #random flot in [0,1), which is fine

                print("After randomizing ", avgPredSeg.take(disQidxs) )
                
            
            if randomizePositions_b == 1: 
                
                #get indices for this chromo and segment:
                idxs = positionsDict[genomeIdName][i]
                print("I'll randomize at ", idxs)
                
                print("before randomizing ", avgPredSeg.take(idxs))
                #randomize:
                for idx in idxs:
    
                    intervalStart = max(idx - randomizePositionsWindow, 0)
                    intervalEnd = min(idx + randomizePositionsWindow, segmentLength)
                    try:
                        if randomizingByShuffle_b == 1:
                            for j in range(intervalStart, intervalEnd+1):
                                jRnd = np.random.randint(0, L, 1)[0] #pick a random index not above the length of the avgPredSeg -1 
                                avgPredSeg[j] = avgPredSeg[jRnd]
                        elif randomizingMultiplicative_b == 1:
                            for j in range(intervalStart, intervalEnd+1):
                                avgPredSeg[j] = np.random.random(1)*avgPredSeg[j]*randomizationScale
                        else:
                            for j in range(intervalStart, intervalEnd+1):
                                avgPredSeg[j] = np.random.random(1)*randomizationScale
                    except IndexError:
                        print("When randomizing positions: Index error at idx", idx)
                        continue
                        
                print("after randomizing ", avgPredSeg.take(idxs))
                        
            
            if randomizeAnnoIntervals_b == 1:
                
                 avgPredSeg = displaceIntervalsInSegments(inputArray = avgPredSeg.copy(), intervalsDict = randomizeAnnoIntervalsDict, chromoName = genomeIdName, segmentId = i )


            data_b = 1
        
        except IOError:
            
            if plots_b == 1:            
                #save the "hanging"  coeff's plot:
                if forATorGCbias_b == 0:
    
                    if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput + 'Fourier_raw_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                    
                    elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput + 'Fourier_raw_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
     
                    elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
    
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput + 'Fourier_raw_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput + 'Fourier_raw_end_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput + 'Fourier_raw_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput + 'Fourier_raw_end_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    


                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput + 'Fourier_raw_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput + 'Fourier_raw_end_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                       
    
                
                elif forATorGCbias_b == 1:
    
                    if shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        
                        
                    elif shuffle_b == 1 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
                        
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                    elif shuffle_b == 0 and randomizeDisqualified_b == 1 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 0:
    
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizedDisqualified_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedDisqualified_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
    
                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 1 and randomizeAnnoIntervals_b == 0:
    
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizedPositions_' + randomizePositionsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizedPositions_' + randomizePositionsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    

                    elif shuffle_b == 0 and randomizeDisqualified_b == 0 and randomizePositions_b == 0 and randomizeAnnoIntervals_b == 1:
    
                        if plotOnlyNorm_b == 0:
                            fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig1_loc.savefig(rootOutput_forATorGCbias + 'Fourier_maxToEnd_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                            fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
    
                        fig2_loc.savefig(rootOutput_forATorGCbias + 'Fourier_norm_randomizeAnnoIntervals_' + randomizeAnnoIntervalsName + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                        
            return
        
        cnt += 1
        
    
    
def inverseFourierOnSegments(rootOutput,
                             segmentLength,
                             genomeIdName, #e.g. hg38_chr17
                             averageRevComplementary_b,
                             dumpFourier_b = 0,
                             peaksOnly_b = 1,
                             peaksList = [],
                             peakLength = 1000,
                             nrSegments = 1000, #just set this to a high number above the true number of segments
                             segStart = 0,
                             inputArrayType = 0,
                             ratioQcutoff = 0.9,
                             windowLength = 1,
                             stepSize = 1,
                             fourierStart = 1, 
                             fourierStop = 15000,
                             fourierWindowLength = 500,
                             fourierStep = 100,
                             forATorGCbias_b = 0,
                             rootOutput_forATorGCbias = '',
                             fileNamePrefix = '',
                             plots_b = 1, 
                             plotFrq = 10,
                             randomPlotPos_b = 0,
                             plotPosition = 500000,
                             plotIntervalLength = 100000,
                             plotStep = 100,
                             saveAtDpi = 300
                             ):
    '''
    Computes the inverse Fourier of segment arrays placed in rootOutput, these input segment
    arrays are assumed to be created by the fct computeFourierOnSegments through dumping the 
    'raw Fouriers'.
    
    If fileNamePrefix is left at default value, the fct returns the inverse Fourier transforms
    of these input arrays.
    
    peaksOnly_b: usually we only want to get the inverse transf of certain peaks
    peaksList: list of top-points of the desired peaks
    peakLength: length of peaks
    
    Else, if fileNamePrefix is set (say to MODIFIED), the fct will return the difference of 
    the inverse Fouriers of the set of segment arrays and their modified counterparts. Ex:
    say that we have Fourier transformed an array of model probabilities, and that we have
    obtained the Fourier transforms of the same arrays modified by randomizing at every 
    position considered a repeat (both obtained by computeFourierOnSegments). Suppose the 
    Fourier transforms of the modified arrays are named with MODIFIED as prefix followed 
    by the name of the Fourier tr's of the unmodified arrays (Fourier_NN, say). Then the fct 
    returns

    inverse Fourier( Fourier_NN - MODIFIED_Fourier_NN )
    
    (for each of the segments implicitly given by the input Fourier arrays).
    
    The motivation for this: by randomizing at all repeats it seems that all 'periodic signals'
    disappear; only the 1/frq type of decay is retained. By subtracting this 'periodicity less' part
    from the whole, the hope is that (almost) only the periodic part remains. On the back of this
    one can then chose a set of peaks to invert.
    '''

    if plots_b == 1:
        fig0, ax0 = plt.subplots()
        
    if len(fileNamePrefix) == 0:
        fileNamePrefixOut = 'None'
    else:
        fileNamePrefixOut = fileNamePrefix
    
       
    rootInput_chr = rootOutput + genomeIdName + r"/"
    
    if inputArrayType == 0:
        rootOutput_chr = rootInput_chr + "FourierOnPredReturn/" + str(fourierStart) + 'to' + str(fourierStop)  + "/"
        
        #only in use if forATorGCbias_b = 1:
        rootOutput_forATorGCbias_chr = rootOutput_forATorGCbias + genomeIdName + "/" + "FourierOnPredReturn/" + str(fourierStart) + 'to' + str(fourierStop) + "/"
    
    
    elif inputArrayType == 1:
        rootOutput_chr = rootInput_chr + "FourierOnRefBaseProb/" + str(fourierStart) + 'to' + str(fourierStop)  + "/"
        
        #only in use if forATorGCbias_b = 1:
        rootOutput_forATorGCbias_chr = rootOutput_forATorGCbias + genomeIdName + "/" + "FourierOnRefBaseProb/" + str(fourierStart) + 'to' + str(fourierStop) + "/"
      

    cnt = 0
    for  i in range(nrSegments):
    
        if forATorGCbias_b == 0:
            loadFile = rootOutput_chr + 'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg'  + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)


            try:
                fftArray = pickle.load(open(loadFile, "rb"))
            except IOError:
                print("File of Fouriers not found for segment %d so breaks here" % cnt)
                cnt += 1
                continue
            if len(fileNamePrefix) > 0:
                loadFileBase = rootOutput_chr + fileNamePrefix + 'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
    
                try:
                    fftArrayBase = pickle.load(open(loadFileBase, "rb"))
                except IOError:
                    print("File of Fouriers with the provided prefix not found for segment %d so breaks here" % cnt)
                    cnt += 1
                    continue

                
                fftArrayDiff = fftArray - fftArrayBase
                
            else:
                fftArrayDiff = fftArray
        
        elif forATorGCbias_b == 1:
            loadFile = rootOutput_forATorGCbias_chr  +  'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' +  str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
            try:
                fftArray = pickle.load(open(loadFile, "rb"))
            except IOError:
                print("File of Fouriers not found for segment %d so breaks here" % cnt )
                cnt += 1
                continue
    
            if len(fileNamePrefix) > 0:
                loadFileBase = rootOutput_forATorGCbias_chr + fileNamePrefix + 'FourierTransform_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
                try:
                    fftArrayBase = pickle.load(open(loadFileBase, "rb"))
                except IOError:
                    print("File of Fouriers with the provided prefix not found for segment %d so breaks here" % cnt)
                    cnt += 1
                    continue
                
                fftArrayDiff = fftArray - fftArrayBase
                
            else:
                fftArrayDiff = fftArray
    
        #To only get the inverse transform of the peaks in the peaksList:
        if peaksOnly_b == 1:        

            x = np.zeros(shape = fftArrayDiff.shape) #array to hold Fourier coeff's of the peaks
            #get the coeff's for each frq in a peak and its negative companion (we want a real valued inverse) 
            for nPeak in peaksList: 
#                print(fftArrayDiff[nPeak:(nPeak+100)])
                x[nPeak:(nPeak+peakLength)] = fftArrayDiff[nPeak:(nPeak+peakLength)]
                x[-(nPeak+peakLength -1 ):(-nPeak + 1)] = fftArrayDiff[-(nPeak+peakLength -1):(-nPeak +1)]
#                print(x[nPeak:(nPeak+10)], x[-(nPeak+10 -1):(-nPeak +1)] )
            invFft = ifft(x)
            #dump it:
            if forATorGCbias_b == 0:
                dumpFile = rootOutput_chr + 'InvFourierTransform_Peak_raw_vs_' + fileNamePrefixOut + str(peaksList) + '_' + str(peakLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
            elif forATorGCbias_b == 1:
                dumpFile = rootOutput_forATorGCbias_chr + 'InvFourierTransform_Peak_raw_vs_' + fileNamePrefixOut + str(peaksList) + '_' + str(peakLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment'  + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
            
            if dumpFourier_b == 1:
                pickle.dump(invFft, open(dumpFile, "wb"))
    
    
        else: #inverse transform all
            
            #Compute the inverse of the fftArrayDiff and dump the result:
            invFft = ifft(fftArrayDiff)
            #dump it:
            if forATorGCbias_b == 0:
                dumpFile = rootOutput_chr + 'InvFourierTransform_raw_vs_' + fileNamePrefixOut + '_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
            elif forATorGCbias_b == 1:
                dumpFile = rootOutput_forATorGCbias_chr+ 'InvFourierTransform_raw_vs_' + fileNamePrefixOut + '_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segment'  + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)
            
            if dumpFourier_b == 1:
                pickle.dump(invFft, open(dumpFile, "wb"))
        
        
        if plots_b == 1:
            
            if randomPlotPos_b == 1:
                #get a random position in the segment from which to start the plot:
                plotPos = np.random.randint(low=0,high=fftArrayDiff.shape[0])
            else:
                plotPos = plotPosition
            
            Xs = np.arange(start = plotPos, stop = plotPos + plotIntervalLength, step = plotStep)
            Ys = [100*np.sum(invFft[n:(n+plotStep)]) for n in Xs]
       
    #        ax0.plot(Xs, Ys, linestyle = lineStyle, alpha = ratioQ, label = str(i) + ' , Q: ' + ratioQlabel)
            ax0.plot(Xs, Ys,  label = str(i))
    
            if (cnt+1)%plotFrq == 0:
                
                if forATorGCbias_b == 0:
                    fig0.savefig(rootOutput_chr + 'Length_' + str(plotIntervalLength) + '_interval_of_InvFourierTransform_raw_vs_' + fileNamePrefixOut  + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    
                elif forATorGCbias_b == 1:
                    fig0.savefig(rootOutput_forATorGCbias_chr + 'Length_' + str(plotIntervalLength) + '_interval_of_InvFourierTransform_raw_vs_' + fileNamePrefixOut  + '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_'  + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf', dpi = saveAtDpi)    


                #new fig
                segStart = cnt + 1
                fig0, ax0 = plt.subplots()
        
        cnt += 1

   


def computeFourierChromosomes(chromosomeOrderList, 
                         rootOutput,
                         modelFileName, 
                         segmentLength,
                         averageRevComplementary_b,
                         inputArrayType = 0,
                         windowLength = 1,
                         stepSize = 1,
                         ratioQcutoff = 0.9,
                         fourierWindowLength = 500,
                         fourierStart = 1,
                         fourierStop = 15000,
                         fourierStep = 100,
                         fourierRawPlotFrq = 10,
                         plotOnlyNorm_b = 1, 
                         getInvPeak_b  = 0,
                         nPeak = 4500,
                         shuffle_b = 0,
                         randomizeDisqualified_b = 0,
                         randomizePositions_b = 0, #randomize a window around each position specified in positionsDict 
                         randomizingByShuffle_b = 0,
                         randomizingMultiplicative_b = 0,
                         randomizationScale = 1.,
                         positionsDict = {},
                         randomizeAnnoIntervals_b = 0,
                         randomizeAnnoIntervalsName = '',
                         randomizeAnnoIntervalsDict = {},
                         randomizePositionsWindow = 5000,
                         randomizePositionsName = 'RandTSS',
                         forATorGCbias_b = 0,
                         rootOutput_forATorGCbias = '',
                         rootInput_forATorGCbias = '',
                         modelFileName_forATorGCbias = '',
                         fullGC_b = 0,
                         dumpFourier_b = 0,
                         dumpFileNamePrefix = '', #if dumping the Fourier coeffs for some modified version of the input arrays, set this prefix to reveal the content/modification
                         plots_b = 1, #if 0 no plots will be made and saved
                         saveAtDpi = 100):
    '''Compute the Fourier transform on segments of (average) predictions by applying
    computeFourierOnSegments to the data of each chromosome.'''
    
    if len(rootInput_forATorGCbias) == 0:
        rootInput_forATorGCbias = rootOutput_forATorGCbias
    
    for chromoName in chromosomeOrderList:    

        genomeIdName = chromoName #+ '_seg' + str(int(segmentLength))

        rootInput_chr = rootOutput + chromoName + r"/"
        
        if inputArrayType == 0:
            rootOutput_chr = rootInput_chr + "FourierOnPredReturn/" + str(fourierStart) + 'to' + str(fourierStop)  + "/"
            
            #only in use if forATorGCbias_b = 1:
            rootInput_forATorGCbias_chr = rootInput_forATorGCbias + chromoName + "/" 
            rootOutput_forATorGCbias_chr = rootOutput_forATorGCbias + chromoName + "/" + "FourierOnPredReturn/" + str(fourierStart) + 'to' + str(fourierStop) + "/"
        
        
        elif inputArrayType == 1:
            rootOutput_chr = rootInput_chr + "FourierOnRefBaseProb/" + str(fourierStart) + 'to' + str(fourierStop)  + "/"
            
            #only in use if forATorGCbias_b = 1:
            rootInput_forATorGCbias_chr = rootInput_forATorGCbias + chromoName + "/"
            rootOutput_forATorGCbias_chr = rootOutput_forATorGCbias + chromoName + "/" + "FourierOnRefBaseProb/" + str(fourierStart) + 'to' + str(fourierStop) + "/"
        

        if (forATorGCbias_b == 0 and not(os.path.exists(rootOutput_chr))):

            os.makedirs(rootOutput_chr)
            print("Directory " + rootOutput_chr + " created. Output will be placed there.")

        elif (forATorGCbias_b != 0 and not(os.path.exists(rootOutput_forATorGCbias_chr))):
            
            os.makedirs(rootOutput_forATorGCbias_chr)
            print("Directory " + rootOutput_forATorGCbias_chr + " created. Output will be placed there.")
        
#        avgPredSeg, cntCorr_seg, cntTot_seg = computeFourierOnSegments(rootOutput = rootOutput_chr,
        computeFourierOnSegments(rootInput = rootInput_chr, 
                                 rootOutput = rootOutput_chr,
                                     modelFileName = modelFileName,  
                                     segmentLength = segmentLength,
                                     genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                                     inputArrayType = inputArrayType,
                                     averageRevComplementary_b = averageRevComplementary_b,
                                     windowLength = windowLength,
                                     stepSize = stepSize,
                                     ratioQcutoff = ratioQcutoff,
                                     fourierWindowLength = fourierWindowLength, 
                                     fourierStep = fourierStep,
                                     fourierStart = fourierStart,
                                     fourierStop = fourierStop,
                                     fourierRawPlotFrq = fourierRawPlotFrq,
                                     plotOnlyNorm_b = plotOnlyNorm_b,
                                     getInvPeak_b = getInvPeak_b,
                                     nPeak = nPeak,
                                     shuffle_b = shuffle_b,
                                     randomizeDisqualified_b = randomizeDisqualified_b,
                                     randomizePositions_b = randomizePositions_b,
                                     randomizingByShuffle_b = randomizingByShuffle_b,
                                     randomizingMultiplicative_b = randomizingMultiplicative_b,
                                     randomizationScale = randomizationScale,
                                     positionsDict = positionsDict,
                                     randomizeAnnoIntervals_b = randomizeAnnoIntervals_b,
                                     randomizeAnnoIntervalsName = randomizeAnnoIntervalsName,
                                     randomizeAnnoIntervalsDict = randomizeAnnoIntervalsDict,
                                     randomizePositionsWindow = randomizePositionsWindow,
                                     randomizePositionsName = randomizePositionsName,
                                     forATorGCbias_b = forATorGCbias_b,
                                     rootInput_forATorGCbias = rootInput_forATorGCbias_chr,
                                     rootOutput_forATorGCbias = rootOutput_forATorGCbias_chr,
                                     modelFileName_forATorGCbias =  modelFileName_forATorGCbias,
                                     fullGC_b = fullGC_b,
                                     dumpFourier_b = dumpFourier_b,
                                     dumpFileNamePrefix = dumpFileNamePrefix,
                                     plots_b = plots_b,
                                     saveAtDpi = saveAtDpi)
                                     
#        return avgPredSeg, cntCorr_seg, cntTot_seg
                        
    


def computeFourierRandomSegments(rootOutput,
                         modelFileName,                        
                         inputArray,
                         labelArray,
                         qualArray,
                         inputArrayType,
                         genomeIdName, #can be anything, just an id of the inputArray
                         windowLength, #if inputArray is arrived at by a sliding-window approach 
                         stepSize, #stride if inputArray is arrived at by a sliding-window approach 
                         nrSamples,
                         averageRevComplementary_b,
                         fourierParamList = [[1,20000, 100, 1000, 1000000]], #[[start, stop, step, windowLenght, segmentLength]]
                         fourierRawPlotFrq = 10,
                         shuffle_b = 0,
                         forATorGCbias_b = 0,
                         rootOutput_forATorGCbias = '',
                         modelFileName_forATorGCbias = '',
                         dumpFourier_b = 0, 
                         ratioQcutoff = 0.9,
                         saveAtDpi = 100):
    '''Compute the Fourier transform on randomly picked segments of the input
    array (typically the array of predictions on a full chromosome, had by assembling
    the segmented predictions for that chromo).
    
    inputArrayType: if 1 input an array of type predArray (ie array of model's prob's for the four bases at each position)
                    if != 1 input array of type predReturn (any array of real numbers)'''
    
            
    L  = len(inputArray)
    
    if inputArrayType == 1:
        
        #Get the array of ref-base probalities acc to the model:
        avgPred = np.zeros(shape = L, dtype = 'float64') #name avgPred is 'historic'
        for l in range(L):
            avgPred[l] = np.dot(inputArray[l],labelArray[l])
            
    else: 
        
        avgPred = inputArray #just use the inputArray as is; intention: for inputArrayType != 0 input is array ofreal numbers
    
    for params in fourierParamList:
        
        fourierStart, fourierStop, fourierStep, fourierWindowLength, segmentLength = params
                    
        fig0, ax0 = plt.subplots()
        fig2, ax2 = plt.subplots()
        
        for i in range(nrSamples):
        
            sumQual = -1
            thresholdQ = ratioQcutoff*segmentLength
            while sumQual <= thresholdQ:
                
                #fetch a random segment in the array of length segmentLength
                startPos = np.random.randint(0, L-segmentLength, 1)[0]
                endPos = startPos + segmentLength
                #check if the quality is ok:
                qual = np.sum(qualArray[startPos:endPos])
                
                if qual > thresholdQ:
                    sumQual = qual
                    takeArray = avgPred[startPos:endPos]
            
               
            #Fourier transform it:
            fftAvgPred = fft(takeArray) #scipy fast Fourier transform; input is real so we really need only the rfft (no need to do the fft), but let's leave this for later ... 
            N =  fftAvgPred.size
            print("Fourier transf lenght: ", N)
            
            if dumpFourier_b == 1:
                
                if forATorGCbias_b == 0:
                    dumpFile = rootOutput + 'FourierTransform_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_segLength_' + str(segmentLength) + '_segStartAt_' + str(startPos) + '_segEndAt_' + str(endPos)
                    pickle.dump(fftAvgPred, dumpFile)
                
                elif forATorGCbias_b == 1:
                    dumpFile = rootOutput_forATorGCbias + 'FourierTransform_' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + + '_segLength_' + str(segmentLength) + '_segStartAt_' + str(startPos) + '_segEndAt_' + str(endPos)
                    pickle.dump(fftAvgPred, dumpFile)
            
            frqs = np.fft.fftfreq(fftAvgPred.size)
                        
    #     
            stop = fourierStop
            start = fourierStart
##         
#            print("First 10 (positive) frqs: " ,frqs[:10])
#            print("Largest (positive) frq: %f" % frqs[N/2])
#            print("First 10 (negative) frqs: " ,frqs[(N-10):N][::-1])
#            print("Up to stop (%d) with frq %f, the max (min) modulus of Fourier coeff is: %f (%f)" % (stop, frqs[stop], np.max(np.abs(fftAvgPred[:stop])), np.min(np.abs(fftAvgPred[:stop]))) )
#            print("After stop (%d) the max (min) modulus of Fourier coeff is: %f (%f)" % (stop, np.max(np.abs(fftAvgPred[stop:(N/2)])), np.min(np.abs(fftAvgPred[stop:(N/2)]))) )
#            

            Xs = np.arange(start = start, stop = stop, step = fourierStep)
            YAbs = [np.abs(fftAvgPred[n]) for n in Xs]
            yMaxs = [np.max(np.abs(fftAvgPred[n:stop])) for n in Xs]
    
            yNorm = [np.linalg.norm(fftAvgPred[n:(n+fourierWindowLength)]) for n in Xs]
              
    
            #fourier coeff's plot:
            ax0.plot(Xs, YAbs, label = 'Start: ' + str(startPos) + ',End: '+ str(endPos) + 'L: ' + str(segmentLength) + ', W: ' + str(fourierWindowLength) )
            ax0.legend(loc = 'upper right', fontsize = 'small')
            
            #local coeff's norm plot
            ax2.plot(Xs, yNorm, label = 'Start: ' + str(startPos) + ',End: '+ str(endPos) + 'L: ' + str(segmentLength) + ', W: ' + str(fourierWindowLength))
            ax2.legend(loc = 'upper right', fontsize = 'small')
             
        if forATorGCbias_b == 0:

            if shuffle_b == 0:
                
                fig0.savefig(rootOutput + 'Fourier_raw_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) + '.pdf', dpi = saveAtDpi)    
                fig2.savefig(rootOutput + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) + '.pdf', dpi = saveAtDpi)
#                    fig3.savefig(rootOutput + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')    
#                    fig4.savefig(rootOutput + 'InvFourier_peak_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')                
            
            elif shuffle_b == 1:
                
                fig0.savefig(rootOutput + 'Fourier_raw_shuffled_'  + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) + '.pdf', dpi = saveAtDpi )
                fig2.savefig(rootOutput + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength)  + '.pdf', dpi = saveAtDpi)
#                    fig3.savefig(rootOutput + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')
#                    fig4.savefig(rootOutput + 'InvFourier_peak_shuffled_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')
        
        
        elif forATorGCbias_b == 1:

            if shuffle_b == 0:
                
                fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) +  '.pdf', dpi = saveAtDpi)
                fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) +  '.pdf', dpi = saveAtDpi)
#                    fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')                                        
#                    fig4.savefig(rootOutput_forATorGCbias + 'InvFourier_peak_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')

            elif shuffle_b == 1:
                
                fig0.savefig(rootOutput_forATorGCbias + 'Fourier_raw_shuffled_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) +  '.pdf', dpi = saveAtDpi)
                fig2.savefig(rootOutput_forATorGCbias + 'Fourier_norm_shuffled_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '_segLength' + str(segmentLength) + '.pdf', dpi = saveAtDpi)
#                    fig3.savefig(rootOutput_forATorGCbias + 'Fourier_raw_end_shuffled_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')    
#                    fig4.savefig(rootOutput_forATorGCbias + 'InvFourier_peak_shuffled_at' + str(nPeak)+ '_w' + str(fourierWindowLength) + '_step' + str(fourierStep) + '_' + genomeIdName + '_seg' + str(segStart) + '_' + str(cnt) + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf')
 

    
    
    
    
def writePredictionsToFile(startAtPosition, 
                           endAtPosition, 
                           predArray, 
                           qualArray, 
                           sampledPositions, 
                           sampledPositionsBoolean):

    pass                            
                            


def loadAnnotation(rootAnnotationFiles, chromosomeOrderList, chromosomeDict , chromosomeLengthFile, firstChromoNr, annotationName, rootOutput, buildAnnotationIntervalDict_b = 0, fromYuhusFolder_b = 0 , organism = 'human', otherRowIdx1 = 1, otherRowIdx2 = 2, extension = ''):
    '''For loading content of bed-file into binary array (which the function dumps 
    with cPickle. Also, if wanted (buildAnnotationIntervalDict_b = 1), a dictionary
    mapping an index of each annotation interval (eg the start and end positions of 
    a repeat) to the (start, end) of the interval can be dumped (the index is just
    a running numbering of the annotated segments,eg repeats).
    
    Possible annotationName's:
    rmskRepeat
    simpleRepeat
    
    repeat (rmsk; repeat masker)
    simple_repeat (tandem repeats; trf)
    3UTR
    5UTR
    introns
    cds
    genes
    
    Possible subType's:
    LINEs, SINEs and their subtypes.
    These can be had from Yuhu's downloads by grepping and placing the output in
    your own folder; then use organism = other with the default otherRowIdx1/2 
    (values as when reading Yuhu's downloads) or custom value.
    
    '''
    
    i = firstChromoNr + 1

    if fromYuhusFolder_b:   
        
        rowIdx1 = 1
        rowIdx2 = 2
    
    else:
        
        if organism == 'human':
            
            rowIdx1 = 6
            rowIdx2 = 7
            
        elif organism == 'yeast':
 
            rowIdx1 = 2
            rowIdx2 = 3 

        elif organism == 'other':
            
            rowIdx1 = otherRowIdx1
            rowIdx2 = otherRowIdx2

    #We need the length of each chromosome; stored here:
    chromoLengthDict = pickle.load(open(chromosomeLengthFile,"rb"))
        
    for chromoName in chromosomeOrderList:
                
        chromoLength = chromoLengthDict[chromoName]

        #typically like 'path/23.CHR22/UCSC/chr22_repeat.bed'
        if fromYuhusFolder_b: 
            if i > 9:
                annotationFileId = str(i) + '.CHR' + str(i-1) + '/UCSC/' + chromoName[5:] + '_'  + annotationName + '.bed'
            else:
                annotationFileId = '0' + str(i) + '.CHR' + str(i-1) + '/UCSC/' + chromoName[5:] + '_'  + annotationName + '.bed'
                
        else:
            annotationFileId = chromoName + '_' + annotationName + extension
            
        fileName = rootAnnotationFiles + annotationFileId
 
#        startAtPosition, endAtPosition = chromosomeDict[chromoName]
        
        kLast = 0
        annotationList = []
        annotationIntervalDict = {}
        annoIdx = 0
        try:
            of = open(fileName, mode='r')
        except IOError:
            print("File: %s not found" % fileName)
            i -= 1
            continue
            
        with of as bedFile:
            
            bedReader = csv.reader(bedFile, delimiter='\t')
            
            lineCnt =  0
            for row in bedReader:
                
                if lineCnt < 1:
                    lineCnt += 1
                    print(row)
                    continue
                
                annoStart = int(row[rowIdx1]) - 1 #subtract 1 since positions in bed files start at index 1
                annoEnd = int(row[rowIdx2]) -1 #ditto
                
                if lineCnt < 5:
                    print(row)
                    print(annoStart, annoEnd)
                    
                annotationIntervalDict[annoIdx] = (annoStart, annoEnd)
                annoIdx += 1
                    
                for k in range(kLast, annoStart): #is the intervals are overlapping we may have annoStart < kLast, but it aht case therange is empty
                    
                    annotationList.append(0)
                
                newStart = max(kLast, annoStart)
                for k in range(newStart, annoEnd +1 ):
                    
                    annotationList.append(1)
                    
                kLast = k +1 #should then be eq to annoEnd +1 
                    
                
#                if annoEnd >= startAtPosition:
#                    
#                    if annoStart >= startAtPosition:
#                        
#                        for j in range(annoStart, annoEnd + 1):
#                            
#                            binArray[j] = 1
#                            
#                    else:
#                        
#                        for j in range(startAtPosition, annoEnd + 1):
#                            
#                            binArray[j] = 1
                
                
                lineCnt += 1
        
        #fill in 0's from kLast to end:
        for k in range(kLast, chromoLength): #last idx is then chromoLength -1
            annotationList.append(0)
        
                
        #Convert the list to an array ad dump it:
        binArray = np.asarray(annotationList, dtype = 'int8')
                           
        if binArray.shape[0] != chromoLength:
            print("Warning! the generated binary array for annotation %s is diff from chromoLength" % annotationName )
        
        dumpFile = rootOutput + chromoName + '_annotationArray_' + annotationName
        pickle.dump(binArray, open(dumpFile, "wb") )
        
        if buildAnnotationIntervalDict_b == 1:
            
            dumpFile = rootOutput + chromoName + '_annotationIntervalDict_' + annotationName
            pickle.dump(annotationIntervalDict, open(dumpFile, "wb") )
    
            
        bedFile.close()
        
        print("In chr: %s Total %s annotated positions %d of a total of %d " % (chromoName, annotationName, np.sum(binArray), binArray.shape[0]))
        
        i -= 1


#util fct to establish an annotation array for a combination of two annotations.
#Aimed for combining repeat and simple_repeat
def combineAnnotations(chromosomeOrderList, chromosomeDict , chromosomeLengthFile, firstChromoNr, annotationName1, annotationName2, annotationNameCombined, rootOutput, organism = 'human', combinationType = 'union'):
    '''Possible combinationType's: union, intersection. Annotations are supposed to be
    in the guise of boolean arrays.'''

    for chromoName in chromosomeOrderList:
        
        passed_b = 0 #just an indicator helping to combine the repeat and simple_repeat bed's

         #load the anno arrays:
        annotationFile1 = rootOutput + chromoName + '_annotationArray_' + annotationName1
        annotationFile2 = rootOutput + chromoName + '_annotationArray_' + annotationName2
        try:
            
            annotationArray1 = pickle.load(open(annotationFile1,"rb"))
            passed_b +=1
            
        except IOError:
            
            print("Annotation file %s not found" %  annotationFile1)
            continue

        try:

            annotationArray2 = pickle.load(open(annotationFile2,"rb"))
            passed_b +=1            

        except IOError:
            
            print("Annotation file %s not found" %  annotationFile2)
            continue

            
        #generate the binArray for the combined anno:
        if passed_b == 2:
            
            if combinationType == 'union':  
                binArrayComb = np.maximum(annotationArray1, annotationArray2)
            elif combinationType == 'intersection':
                binArrayComb = np.minimum(annotationArray1, annotationArray2)
            
            #dump  it
            dumpFile = rootOutput + chromoName + '_annotationArray_' + annotationNameCombined
            pickle.dump(binArrayComb, open(dumpFile, "wb") )
 
    

#util for obtaining a new set of annotation intervals from an existing dict of intervals for some
#annotation, by randomly displacing the intervals. Aimed at the Fourier transf:
def rndDisplaceAnnoIntervalsInSegments(rootOutput, chromoName, annotationName, segmentLength, firstSegmentStart, rndPct):
    '''Returns a dict mapping each interval's idx to its original (start,end) positions and
    a pair of new such positions.
    Only (about) rndPct pct of the intervals are displaced.'''


    loadFile = rootOutput + chromoName + '_annotationIntervalDict_' + annotationName
    intervalsDict = pickle.load( open(loadFile, "rb") )
    
    intervalIdxs = intervalsDict.keys()
    intervalIdxs.sort()
    
#    print(intervalIdxs)

    #find the number of segments needed to be covered:    
    lastIdx = max(intervalIdxs)
    print("Last idx: ", lastIdx)
    lastAnnoStart, lastAnnoEnd = intervalsDict[lastIdx]
    endPos =  segmentLength
    n = 0
    while endPos < lastAnnoEnd: 
        endPos += segmentLength
        n +=1
    
    nrSegments = n
    
    outDict = {}
    outDict[chromoName] = {}
    idx = 0
    prevStart, prevEnd = firstSegmentStart, firstSegmentStart
    fractionRnd = float(rndPct)/100.
    cntValueError = 0
    for i in range(nrSegments):
        
        if not(i in outDict[chromoName]):
            outDict[chromoName][i] = {}
        
        if i == 0:
            startPos = firstSegmentStart
            endPos =  segmentLength
        else:
            startPos = endPos
            endPos =  startPos + segmentLength

        start,end = intervalsDict[idx]
        while start < startPos:
            idx += 1
            start,end = intervalsDict[idx]
        
        if idx > 0:
            prevStart, prevEnd = intervalsDict[idx - 1]
            
        while end < endPos and idx < lastIdx:
            
            start,end = intervalsDict[idx]

            #decide whether to displace this interval (idx) or not:
            rand_b = np.random.binomial(1,fractionRnd)
            
            if rand_b == 0:
                newStart, newEnd = start, end
                
            else: 
                    
                nextStart, nextEnd = intervalsDict[idx + 1]
               
                if nextStart >= endPos:  
                    nextStart = endPos
                    
                if prevEnd < startPos:
                    prevEnd = startPos
            
                #pick a random position bewteen the previous (new) end and the next start:
                L = end - start
                try:
                    newStart = np.random.randint(prevEnd, nextStart - L)
                except ValueError:
                    print(idx, prevEnd, start, end, nextStart, L)
                    cntValueError += 1
                    newStart = start
                newEnd = newStart + L
    
            #update
            prevStart, prevEnd = newStart, newEnd
                
            outDict[chromoName][i][idx] = [(start - i*segmentLength,end - i*segmentLength), (newStart - i*segmentLength, newEnd - i*segmentLength)]

            idx += 1


    print("There were %d cases with no space for changing interval position (ValueError)" % cntValueError   ) 
    
    dumpFile = rootOutput + chromoName + '_annotationIntervalDict_' + annotationName + '_rndDisplacedAt' + str(rndPct) + 'Pct' 
    pickle.dump(outDict, open(dumpFile, "wb") )
    
    return outDict 


#fct for making a bed file from some ctcf-annotation files (from paper by Fu et al
#The Insulator Binding Protein CTCF Positions 20 Nucleosomes around Its Binding Sites across the Human Genome):
#From the bed-file a annotation file for each chromo can be obtained simply by grepping; from there
#the loadAnnotation fct can be applied.
def bedFileFromCTCFAnno(inputFileName, outputFileName , pattern = '(chr[X Y 0-9]+):([0-9]+)-([0-9]+)'):
    
    #flush output file
    outputFile = open(outputFileName, 'w')
    outputFile.close()
    #.. and open for writing in append mode:
#    outputFile = open(outputFileName, 'a')
    
    try:
        handle = open(inputFileName)
    except IOError:
        print("File: %s not found, so exits" % inputFileName)
        return
                
    # Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length
    toMatch = re.compile(pattern)
    lineCnt = 0
    with open(outputFileName, 'a') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter='\t')
        
        while True:
            
            lines = handle.readlines(10000) 
            if not lines:
                    break  # no more lines to read
            
            for line in lines:
            
                v = line.strip()
    
                if lineCnt == 0:
                    print("1st line:\n ", line)
                    lineCnt += 1
                    continue
                
                v = v.split()
                print(v)
    #            try:
                if len(v) > 1:
                    print(v[0])
                    print(v[1])
                    try:
                        m = toMatch.match(v[0])
                        chrNN = m.group(1)
                        start = m.group(2)
                        end = m.group(3)
                        strand = v[1]
                        csvWriter.writerow([chrNN,start, end, strand])
                    except AttributeError:
                        pass
                    
                
                lineCnt += 1
                print(lineCnt)
                
            
    handle.close()
    outputFile.close()
        
                
            
                
            
                
            



#def generateRepeatArraysGenomeSeqMasking(rootGenome, chromosomeList, rootOutput):
#    
#
#    #Loop over chr's; call fastReadGenome and get the repeat seq list of 0/1's).    
#    #Convert to array.
#    
#        
#    for chromoName in chromosomeList:
#        
#        fileName = chromoName + ".txt"
#        fileGenome = rootGenome + fileName
#        
#        #Read in data from genome and get it encoded:
#        exonicInfoBinaryFileName = ''
#        chromoNameBound = 100
#        startAtPosition = 0
#        endAtPosition  = 3e9
#        outputAsDict_b = 0
#        outputGenomeString_b = 1 #!!!
#        randomChromo_b = 0
#        avoidChromo = []
#
#
#        Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileName = fileGenome, 
#                   exonicInfoBinaryFileName = exonicInfoBinaryFileName,
#                   chromoNameBound = chromoNameBound, 
#                   startAtPosition = startAtPosition,
#                   endAtPosition = endAtPosition,
#                   outputAsDict_b = outputAsDict_b,
#                   outputGenomeString_b = outputGenomeString_b,
#                   randomChromo_b = randomChromo_b, 
#                   avoidChromo = avoidChromo)
#                   
#        repeatArray = np.asarray(Xrepeat, dtype = 'int8')
#        #dump it
#        dumpFile = rootOutput + chromoName + '_annotationArray_repeatsGenomeSeq'
#        pickle.dump(repeatArray, open(dumpFile, "wb") )

        
    

def getPredArrayFrqModel(getAllPositions_b, samplePositions,  sampleGenomeSequence, labelArray, fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, resultsDictFrqModel, k, samplePositionsIndicatorArray = 0):

    '''
    Obs: the check of the label arrar in the code assumes that the genome sequence was 
    read with fastReadGenome.
    
    Input:
        
    samplePositions: array of the positions of the samples; the samples must be generated with the same data (fileGenome) etc --- same startAtPosition
    and endAtPosition. The sample positions will (then) be relative to the startAtPosition.       
    NOT REALLY NEEDED SO 0 BY DEFAULT: samplePositionsIndicatorArray: array of booleans (0/1) indicating whether the prediction should be called for a position (1) or not (0)     
    sampleGenomeSequence: genome string corr to samplePositions
    labelArray: the array of labels (encoded bases) corr to the sampleGenomeSequence (if getAllPositions_b = 1 both these can be taken to be empty; if getAllPositions_b = 0 the arrays must be provided and correspond)
    getAllPositions_b: if 1 all positions will be sampled ##old rubbish?:the samplePositionIndicatorArray will consist of 1's for the length of the read-in genomic data (which is done by fastReadGenome)
    '''
    
    if getAllPositions_b == 1:

        #Read the genome seq (X is the string in upper case):
        Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileName = fileGenome, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = 0,
#               outputGenomeString_b = 0,
               randomChromo_b = 0, 
               avoidChromo = [])
    
        lenX = len(X)

        
#        lenPred = lenX
#        nrSamples =  lenX - 2*k - 1
        samplePositions = np.arange(k, lenX - k - 1, dtype='int64')
#        for i in range(lenX):
#            samplePositions[i] = i
        
        samplePositionsIndicatorArray = 0 #dummy; this array turned out unnecc; more lines #'ed below
#        samplePositionsIndicatorArray = np.ones(shape=lenX)
#        
#        only the first and last part both of length flankSize are not "sampled":
#        for i in range(k):
#            samplePositionsIndicatorArray[i] = 0 
#            samplePositionsIndicatorArray[lenX - 1 - i] = 0
            
        lenPred = samplePositions.shape[0]
        
    else:
        
        print("getAllPositions_b is set to 0, so an array of samplePositions and corresponding genomeSequence must be provided")
    
        lenPred = samplePositions.shape[0]  #= number of samples (some may be repeated)
        X = sampleGenomeSequence
    
    predArray = np.zeros(shape=(lenPred, 4)) #data type is float64 by default
    Q = np.ones(shape = lenPred, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    corrLabelArray = np.zeros(shape=(lenPred, 4), dtype = 'int8') #label array to be derived from the provided genome seq (if any)
    
    #Loop through the genome seq, get the context at each position and look up the 
    #the predicted distr in the results of the frq model:
    cntMissingKeys = 0
    i = 0 #index in pred/Q array
    for idx in samplePositions: #Obs: lenPred = len(samplePosition); and the indexing of the predArray follows that of the samplePositions
        
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
            #set to random:
            predArray[i] = np.asarray([.25, .25, .25, .25])
#debatable ... the k-mer models ought to cover all possible contexts; 
#not disq'ing has a few implications downstream (eg in LR test)            
#            #and disqualify:
#            Q[i] = 0
            

#        if i < 10:
#            print idx
#            print context
#            print X[idx-k:idx+k+1]
#            print predArray[i]

        
        try:
            #fastReadGenome returns bases in upper-case: 
            if X[idx] == 'A':            
                corrLabelArray[i] = dataGen.codeA_asArray
            elif X[idx] == 'T':     
                corrLabelArray[i] = dataGen.codeT_asArray
            elif X[idx] == 'C':    
                corrLabelArray[i] = dataGen.codeC_asArray
            elif X[idx] == 'G':    
                corrLabelArray[i] = dataGen.codeG_asArray
            else:
                corrLabelArray[i] = dataGen.codeW_asArray #wild-card array

        except IndexError:

            print("No entry in genome seq (X) at pos: ", idx)
            corrLabelArray[i] = dataGen.codeW_asArray #wild-card array
        
        i += 1
            
    #In case we have set getAllPositions_b = 0 and so have derived/obtained the 
    #predArray for the provided sampleGenomeSequence, the provided labelArray
    #should be id to the corrLabelArray that we have just derived:
    if getAllPositions_b == 0:
        
        if np.array_equal(corrLabelArray, labelArray):
            
            print("Fine, the derived label array corr's to the provided labelArray.")
        
        else:
            
            print("No go (if you've used fastReadGenome for reading in the genome seq): the derived label array does NOT corr to the provided labelArray!")
            print(corrLabelArray.shape, labelArray.shape)
            print(corrLabelArray[100000:100005])
            print(labelArray[100000:100005])
            
    print("cntMissingKeys: ", cntMissingKeys)
            
        
    return predArray, Q, samplePositions, samplePositionsIndicatorArray, corrLabelArray
    
    


def loglikelihoodRatioTestNonNestedModels(encodedGenomeData, samplePositions, predArrayFrqModel, qualifiedArrayFrqModel, k, predArrayNN, qualifiedArrayNN, flankSize, useSubSample_b = 1, subSampleSize = 10000000,subSampleFraction = 0.1, dumpResults_b = 0, modelNameFrq = 'frqModel', averageRevComplementaryFrq_b = 0, modelNameNN = 'NNmodel', averageRevComplementaryNN_b = 0, rootOutput = ''):
    
    '''
    Computes the generalized LR test for non-nested models acc to Vuong/White/Cox test.
    
    Input:
        encodedGenomeData: one hot encoded genomic sequence.
#        samplePositionIndicatorArray: array of booleans (0/1) indicating whether a position in the encodedGenomeData is to be considered or not (ie whether it was sampled or not for the predictions)
        
        samplePositions: the sampled positions must be the same for the two models; their qual and pred-arrays must corr to these positions (in
        particular they must have the same length).
        
        predArrayFrqModel: array of predicted distributions (for the occ of the four letters, ACGT) for the same genomic sequence for some model making predictions
        on contexts parameterized by a single parameter, k (eg a k-mer model)
        qualifiedArrayFrqModel: array of booleans (0/1) obtained together with the predArrayFrqModel, indicating whether the position is dis-qualified (0) or not (1), ie if the position contains a not-ACGT or an ACGT.
        predArrayNN: array of predicted distributions (for the occ of the four letters, ACGT) for the same genomic sequence for some model making predictions
        on contexts parameterized by a single parameter, flankSize (eg a NN convolution model taking contexts of flanksize flankSize) 
        qualifiedArrayNN: as qualifiedArrayFrqModel, but for the NN  model
        
        subSampleFraction: the fraction of the sampled positions on which to do the test; the positions considered are randomly
        picked from the samplePositions (0.1).
        
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
    #compute l_NN = log(p_NN(base_i)) (if p_NN(base_i) == 0 set log(...) = -1e20)
    #compute l_frq = log(p_frq(base_i)) (if p_frq(base_i) == 0 set log(...) = -1e20)
    #add LR_i = l_NN - l_LR to LR and LR_i^2 to varLR
    #when loop complete: return LL, varLR
    
    #Obs: 
    #predArrayFrqModel starts at genome position k (the first left-hand flank is at 0,1 ..., k-1)
    #predArrayNN start at genome position flankSize (the first left-hand flank is at 0,1 ..., flankSize-1)
    
    
    if len(encodedGenomeData) != 5:
        
        print("You have probably run encodeGenome (dnaNet_dataGen module) without asking to have the genome sequence included in output. Change and rerun.")
    
    else:
        
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString, chromoList =  encodedGenomeData 

#    
#    lenGenome = genomeSeq.shape[0]
    
    
    print("nr of sampled positions " ,len(samplePositions) )
    print("shape of pred array/qual array, frq model: ",predArrayFrqModel.shape, qualifiedArrayFrqModel.shape )
    print("shape of pred array/qual array, NN model: ",predArrayNN.shape , qualifiedArrayNN.shape )

    startAt = max(flankSize, k)
    print("First position to be considered: ", samplePositions[0], " Should be the same as max of flankSize and k: ", startAt)
    
    if useSubSample_b == 1:
        
        #take out subSampleFraction worth of positions on which to perform the test; we make
        #a binary array to hold the info for convenience:
        fractionSampledArray = np.zeros(shape = len(samplePositions))

        #Use the absolute sub-sample size if provided:
        if subSampleSize > 0:
            
            subSampleFraction = min(float(subSampleSize)/len(samplePositions), 1)
            
        for i in range(len(samplePositions)):
            
            ind = np.random.binomial(1,subSampleFraction)
                   
            if ind == 1:       
                fractionSampledArray[i] = ind


    n = 0
    LR = 0.0
    v = 0.0
#    for i in range(startAt, lenGenome - startAt - 1):
    i = 0 #index in pred/Q arrays
    for idx in samplePositions:
        
        if fractionSampledArray[i] == 0:
            i +=1
            continue
        
        idx = np.int(idx) #necessary ... re-casting the array of positions to 'int64' earler doesn't help (?!) 
        
#        if samplePositionsIndicatorArray[idx] == 0:
#            continue
        
        #Determine whether the position is qualified or not; if not skip the position:
        if qualifiedArrayFrqModel[i] == 0 or qualifiedArrayNN[i] == 0: #np.max(genomeSeq[i]) > 2 or 
            i += 1
            continue
        
        predFrq = predArrayFrqModel[i]
        predNN = predArrayNN[i]
        
#        print(predFrq.shape, predNN.shape, idx, i)
        
        #To get the predicted probability of the base at position i:
        #-- the encoding has a 1 at the index of the base, and zeros elsewhere
        #-- so the dot product of the predicted distr with the encoding gives the probability acc to the model of the actual base
        dot_frq = np.dot(predFrq,genomeSeq[idx]) 
        if dot_frq > 0: 
            l_frq = np.log(dot_frq) 
        else:
            l_frq = -1e20
        dot_NN = np.dot(predNN,genomeSeq[idx])
        if dot_NN > 0:
            l_NN = np.log(dot_NN)
        else:
            l_NN = -1e20
        
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
        
    #If the two models are equiv, the log-likelihood ratios will be normally
    #distr with mean 0 and (some) variance sigma. In that case the mean, LR/n, 
    #will be normally distr with mean 0 and variance sigma/n, with n = nr of 
    #samples. Then: LR/n/sqrt(sigma/n) = LR/sqrt(n*sigma). An estimate of
    #sigma is var computed here: 
    print("n, sample size ", n, np.sum(fractionSampledArray))
    print("The LR is found to be: ", LR)
    var = v/n - np.power(LR/n,2)
    print("The variance is found to be: ", var)
    print("LR/n ", LR/n)
        
    testFig = LR/np.sqrt(n*var) 
    print("The test figure is (LR/sqrt(n*variance)): ", testFig)
    
    #look-up the percentile:
    pVal = 1 - stats.norm.cdf(testFig, loc=0, scale=1)
    
    print("p-value of generalized log-likelihood test: ", pVal)

        
    if dumpResults_b == 1:
        
        #dump results:
        if useSubSample_b == 1:
            dumpFile = rootOutput + 'generalizedLRtest_' + modelNameFrq + '_vs_' + modelNameNN +  '_avgRevCompl1' + str(averageRevComplementaryFrq_b) +  '_avgRevCompl2' + str(averageRevComplementaryNN_b) + '_subSampled' + str(useSubSample_b) + '_subSampleSize' + str(subSampleSize) + '_subSampleFraction' + str(subSampleFraction)
        else:
            dumpFile = rootOutput + 'generalizedLRtest_' + modelNameFrq + '_vs_' + modelNameNN +  '_avgRevCompl1' + str(averageRevComplementaryFrq_b) +  '_avgRevCompl2' + str(averageRevComplementaryNN_b) + '_subSampled' + str(useSubSample_b)

        pickle.dump((pVal, testFig, n, LR, var), open(dumpFile, "wb") )
        
    return pVal, testFig, n, LR, var
      
     

def collectLikelihoodRatioResultsOnChromos(rootResults, chromosomeOrderList, modelName1, averageRevComplementary1_b, modelName2List, averageRevComplementary2_b_list, useSubSample_b, subSampleSize, subSampleFraction, rootOutput, fileName, captionText = 'Results of likelihood ratio tests. Test value is the value of the test statistic and Std dev is the standard deviation of it. Model names are as defined in ....', saveAtDpi = 100):
    '''To gather the results of model1-model2 LR-tests across chromosomes. Makes tex-table of LR-test for totals
    and a plot of test fig-by-chromo across the model2's'''
    
    testFigDictTot = {}
    testFigDictModel = {}
    for modelName2 in modelName2List:
        
        print("Model: ", modelName2)
        
        if not(modelName2 in testFigDictModel):
            testFigDictModel[modelName2] = [] 
            

        #reset/init
        testFig_tot = 0
        n_tot = 0
        v_tot = 0
        LR_tot = 0
       
        for chromo in chromosomeOrderList:
        
            i = 0
            if useSubSample_b == 1:
                loadFile = rootResults + chromo + r'/generalizedLRtest_' + modelName2 + '_vs_' + modelName1 +  '_avgRevCompl1' + str(averageRevComplementary1_b) +  '_avgRevCompl2' + str(averageRevComplementary2_b_list[i]) + '_subSampled' + str(useSubSample_b) + '_subSampleSize' + str(subSampleSize) + '_subSampleFraction' + str(subSampleFraction)
            else:
                loadFile = rootResults + chromo + r'/generalizedLRtest_' + modelName2 + '_vs_' + modelName1 +  '_avgRevCompl1' + str(averageRevComplementary1_b) +  '_avgRevCompl2' + str(averageRevComplementary2_b_list[i]) + '_subSampled' + str(useSubSample_b)
    
            print(loadFile)
            pVal, testFig, n, LR, var = pickle.load(open(loadFile, "rb") )
            
            testFigDictModel[modelName2].append(testFig)

        #Compute the testfig for model modelName2:
        v = n*(var + np.power(LR/n,2))
        v_tot += v
        n_tot += n
        LR_tot += LR
        
        
        print("n ", n_tot)
        print("The LR is found to be: ", LR_tot)
        var_tot = v_tot/n_tot - np.power(LR_tot/n_tot,2)
        print("The variance is found to be: ", var_tot)
        print("LR/n ", LR_tot/n_tot)
            
        testFig_tot = LR_tot/np.sqrt(n_tot*var_tot) 
        print("The test figure is (LR/sqrt(n*variance)): ", testFig_tot)
        
        #look-up the percentile:
        pVal_tot = 1 - stats.norm.cdf(testFig_tot, loc=0, scale=1)
        
        print("p-value of generalized log-likelihood test: ", pVal_tot)
        print("...")
        
        testFigDictTot[modelName2] = testFig_tot, pVal_tot
    
        i += 1 


    #Make tex-string for table of tot results and write it to file:
    s = r'\begin{table}[h!]' + '\n'
    s +=r'  \begin{center}' + '\n'
    s +=r'    \label{tab:table1c}' + '\n'
    s +=r'    \begin{tabular}{l | c | c | c | r} % <-- Alignments: 1st column left, 2nd middle and 3rd right, with vertical lines in between' + '\n'
    s +=r'        \textbf{Model1} & \textbf{Model2} & \textbf{Test value}  & \textbf{p-value}\\' + '\n'        
    s +=r'      \hline' + '\n'
    
    for modelName2 in modelName2List:
        
        testFig_tot, pVal_tot = testFigDictTot[modelName2]
        
        s += modelName1 + r' & ' +  modelName2 + r' & ' + str(testFig_tot) + r' & ' + str(pVal_tot) + r'\\'   + '\n'
    
    s +=r'    \end{tabular}' + '\n'
    s +=r'        \caption{'+ captionText + '}'  + '\n'
    s +=r'  \end{center}' + '\n'
    s +=r' \label{tab:LRtests}' + '\n'
    s +=r' \end{table}'
    
    
    outputFileName = rootOutput + fileName
    outputFile = open(outputFileName, 'w')
    outputFile.write(s)
    outputFile.close()
    
    #Make plot of testfigs over chromos per model:
    colors = cm.get_cmap('Set3')
    fig = plt.figure()
    
    cnt = 0
    for modelName2 in modelName2List: 
    
        yValues = testFigDictModel[modelName2]
        print(chromosomeOrderList)
        print(yValues)


        plt.scatter(chromosomeOrderList, yValues, c = colors(cnt+2), label = modelName2)
        
        cnt += 1
    
    plt.xticks(rotation = 90, fontsize = 'x-small')
    plt.xlabel('Chromosome', fontsize = 'x-small')    
    plt.ylabel('Test figure', fontsize = 'x-small')
    plt.yticks(fontsize = 'x-small')
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = 'x-small') #places legend outside the frame with the upper left placed at coords (1.05, 0.6)

    plt.tight_layout()
    plt.show()
    
    plt.savefig(rootOutput + modelName1 + '_multiLR.pdf', dpi = saveAtDpi)
  
    
      
def compareTwoModelsOnChromosomes(rootGenome, 
                                  chromosomeOrderList, 
                                  chromosomeDict, 
                                  rootOutput, 
                                  rootPredictModel1, 
                                  modelFile1, 
                                  flankSize1, 
                                  modelType1, 
                                  modelName1, 
                                  rootPredictModel2, 
                                  modelFile2, 
                                  flankSize2, 
                                  modelType2, 
                                  modelName2,
                                  firstChromoNr = 1,
                                  frqModelCoverType = 'All chromos', #else: 'Per chromo'
                                  averageRevComplementary2_b = 0, 
                                  segmentLength2 = 1e6, 
                                  averageRevComplementary1_b = 0, 
                                  segmentLength1 = 1e6, 
                                  LRtest_b = 1, 
                                  useSubSampleLR_b =1, 
                                  subSampleSizeLR = 10000000,
                                  subSampleFractionLR = 0.1,
                                  plot_b = 1, 
                                  bins = 50, 
                                  chromoFieldIdx = 5,
                                  annotationTypes = [],
                                  rootAnnotationFiles = '',
                                  nameChangeDict = {},
                                  saveAtDpi = 100):
    
    '''Wrapper of LR tests and plotting of prediction arrays.

    modelType*: Indicates from what source the preictions are had.
                0: internal NN model; 1: internal frq model; 2: external model, Yuhu's folder notation!    

    frqModelCoverType: controls whether results from a frq model (k-mer model) should be 'Per chromo' or not (def: 'All chromos'). 

    '''   
    
    #Step 1: load prediction arrays forboth models and assemble them
    #Step 2: Check that samplePositions are consecutive and take "intersections" of all arrays.
    #Step 3: Compute LR test and plot as wanted
    
    
    #Step 1:

    #Depending on modelType generate dictionaries mapping each key in the chromoDict (chromo name) to the
    #path where to find the predictions
    predictDict1 = {}
    i = firstChromoNr + 1  #only used with model type 2 (AK's Markov, Yuhu's folder structure)
    for chromoName in chromosomeOrderList:
        
        print(chromoName)
        
        if modelType1 == 0:
            
            predictDict1[chromoName] = rootPredictModel1 + chromoName + '/'
            
        elif modelType1 == 1:
              
            if frqModelCoverType == 'Per chromo':  
                #typically like 'path/chr22/frqModel_chr22_k" + str(k) + ".txt"'
                chromoFileId = chromoName[chromoFieldIdx:] + '/frqModel_' + chromoName[chromoFieldIdx:] + '_k' + str(flankSize1) + ".txt" 
            else:
                #typically like 'path/frqModel_k" + str(k) + ".txt"'
                chromoFileId = '/frqModel_k' + str(flankSize1) + ".txt" 

            predictDict1[chromoName] = rootPredictModel1 + chromoFileId            
            
        elif modelType1 == 2:
            
            #typically like 'path/23.CHR22/Chr22.probs'
            if i > 9:
                chromoFileId = str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'
            else:
                chromoFileId = '0' + str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'
                
            i += 1
            predictDict1[chromoName] = rootPredictModel1 + chromoFileId

            
    print("predictDict1 ", predictDict1)
            
            
    #same for second model:
    predictDict2 = {}
    i = firstChromoNr + 1 #only used with model type 2 (AK's Markov, Yuhu's folder structure)
    for chromoName in chromosomeOrderList:
        
        if modelType2 == 0:
            
            predictDict2[chromoName] = rootPredictModel2 + chromoName + '/'
            
        elif modelType2 == 1:
                      
            if frqModelCoverType == 'Per chromo':  
                
                #typically like 'path/chr22/frqModel_chr22_k" + str(k) + ".txt"'
                chromoFileId = chromoName[chromoFieldIdx:] + '/frqModel_' + chromoName[chromoFieldIdx:] + '_k' + str(flankSize2) + ".txt" 

            else:
                
                #typically like 'path/frqModel_k" + str(k) + ".txt"'
                chromoFileId = '/frqModel_k' + str(flankSize2) + ".txt" 


            predictDict2[chromoName] = rootPredictModel2 + chromoFileId            
 
           
        elif modelType2 == 2:
            
            #typically like 'path/23.CHR22/Chr22.probs'
            if i > 9:
                chromoFileId = str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'
            else:
                chromoFileId = '0' + str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'

            i += 1
            predictDict2[chromoName] = rootPredictModel2 + chromoFileId

    print("predictDict2 ", predictDict2)
    
    for chromoName in chromosomeDict:
        
        fileName = chromoName + ".txt"
        fileGenome = rootGenome +fileName
                        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition, endAtPosition = chromosomeDict[chromoName]
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b = 1
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
                       
        #Later we'll call assemblePredictArrayFromSegments, where we need to specify a nr-of-segments; this could eg
        #be set to some high number, eg 1000 but let us do it a little more carefully:
        L = len(encodedGenomeData[0]) #length of genome seq               
        nrSegments1 = int(np.ceil(float(L)/segmentLength1)) + 1 
        nrSegments2 = int(np.ceil(float(L)/segmentLength2)) + 1 

#        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString =  encodedGenomeData 
#        print(encodedGenomeData[3][:100])
        
#        raw_input("S..... er dejlig")


        #Create a directory if its not already there:
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):

            os.makedirs(rootOutput_thisChromo)
            print("Directory " + rootOutput_thisChromo + " created. Output will be placed there.")
    

        #load annotationArray if there:
        annotationDict = {}
        for annoType in annotationTypes:
            
            annotationFile = rootAnnotationFiles + chromoName + '_annotationArray_' + annoType
            
            print(annotationFile)
        
            
            try:
            
                annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))
                
            except IOError:
                
                print("Annotation file %s not found" %  annotationFile)
                continue

        
#        #Get a (sub) set of the encoded data on which to get the models predictions, and, finally, to compute their generalized-LR test figure:
#        genSamplesAtRandom_b = 0 #!!!
#        cutDownNrSamplesTo = 4e9
#        #use these default settings:
#        labelsCodetype = 0
#        outputEncodedType = 'int8'
#        convertToPict_b = 0
#        shuffle_b = 0
#        inner_b = 1
#        shuffleLength = 5
#        augmentWithRevComplementary_b = 0
#
#        
#        flankSize = max(flankSizeNN, flankSizeFrq)
#        outputSamples = dataGen.getAllSamplesFromGenome(encodedGenomeData = encodedGenomeData, genSamplesAtRandom_b = genSamplesAtRandom_b, cutDownNrSamplesTo = cutDownNrSamplesTo, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = flankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)

        #Then outputSamples = X, Y, R, Q, sampledPositions, sampledPositionsBoolean
    
    
        genomeIdName1 = chromoName # + '_seg' + str(int(segmentLength1))    
        genomeIdName2 = chromoName #+ '_seg' + str(int(segmentLength2))  
        ####################################################
        
        #Now get the predictions:
        
        ####################################################
        
        #paths to where to find the predictions:
        rootOutput1 = predictDict1[chromoName] 
        
        rootOutput2 = predictDict2[chromoName]
 
        if modelType1 == 0:
            
            #To obtain a complete prediction array across all segments 
            #(use the same encodedData as when running the prediction across the genome (step1)!):                     
            predArray1, labelArray1, qualArray1, samplePositions1, samplePositionsBoolean1 = assemblePredictArrayFromSegments(rootOutput = rootOutput1, modelFileName = modelFile1, genomeIdName = genomeIdName1, nrSegments= nrSegments1, augmentWithRevComplementary_b = averageRevComplementary1_b, segmentLength = segmentLength1)                     
      
        elif modelType1 == 1:
         
            #temp placeholders
            samplePositions = 0
            samplePositionsIndicatorArray = 0            
            sampleGenomeSequence = ''
            
            resDictFrq = frqM.readResults(rootOutput1)
            getAllPositions_b = 1
            predArray1, qualArray1, samplePositions1, samplePositionsBoolean1, labelArray1 = getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, sampleGenomeSequence = sampleGenomeSequence, labelArray = '', fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel=resDictFrq, k= flankSize1)    
            #Then predArray1, qualArray1, samplePositions1, samplePositionsBoolean1 = predArrayFrq, qualArrayFrq, samplePositions, samplePositionsIndicatorArray 
        
            

        if modelType2 == 0:

            predArray2, labelArray2, qualArray2, samplePositions2, samplePositionBoolean2 = assemblePredictArrayFromSegments(rootOutput = rootOutput2, modelFileName = modelFile2, genomeIdName = genomeIdName2,  nrSegments= nrSegments2, augmentWithRevComplementary_b = averageRevComplementary2_b, segmentLength = segmentLength2)                     
                     
 
        elif modelType2 == 1 or modelType2 == 2:
            
            #For the frq model/external (model2) use the sampledPositions from the NN model (model1):
            getAllPositions_b = 0
            samplePositions = samplePositions1
            labelArray = labelArray1
            samplePositionsIndicatorArray = samplePositionsBoolean1
            sampleGenomeSequence = encodedGenomeData[3] # genomeSeqString is 3rd entry of encodedGenomeData
     

            if modelType2 == 1:
                 
                resDictFrq = frqM.readResults(rootOutput2)
                predArray2, qualArray2, samplePositions2, samplePositionsBoolean2, labelArray2 = getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, sampleGenomeSequence = sampleGenomeSequence, labelArray = labelArray, fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel=resDictFrq, k= flankSize2)    
                #Then: predArray2, qualArray2, samplePositions2, sampledPositionsBoolean2 = predArrayFrq, qualArrayFrq, samplePositions, samplePositionsIndicatorArray

            elif modelType2 == 2:
                
                #Example is the Markov model. There we read in the prob's from Yuhu's run of AK's model. 
                #Read in prob's from external: 
                positionArrayMarkov, predArrayMarkov, refBaseListMarkov = readInProbsFromExternal(rootOutput2)
                #We have to "standardize" these so that the arrays match those from the other model's indexing (here a NN-model).
                displacementOfExternal = 1 #indexing start from 1 in results from Markov model
                predArray2, qualArray2, samplePositions2, labelArray2 = getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence, labelArray =labelArray, positionsExternal= positionArrayMarkov, probsExternal= predArrayMarkov, refBaseListExternal = refBaseListMarkov,  fileGenome =fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= flankSize2, displacementOfExternal = displacementOfExternal)
                #Then predArray2, qualArray2, samplePositions2 = predArrayMarkov_shared, qualArrayMarkov_shared, samplePositionsMarkov_shared  
                


        #Step 2:

        #Check that the samplePositions arrays are "without holes":
        L1 = samplePositions1.shape[0]
        checkDiff1 = samplePositions1[1:] - samplePositions1[:(L1-1)] #should be array of 1's
        check1 =  np.where(checkDiff1 > 1)
        if len(check1[0]) > 1:
            print("Obs! samplePositions1 has %d holes!" % len(check1[0]))
        else:
            print("samplePositions1 has no holes")
        
        L2 = samplePositions2.shape[0]
        checkDiff2 = samplePositions2[1:] - samplePositions2[:(L2-1)] #should be array of 1's
        check2 =  np.where(checkDiff2 > 1)
        if len(check2[0]) > 1:
            print("Obs! samplePositions2 has %d holes!" % len(check2[0]))
        else:
            print("samplePositions2 has no holes")

            
            
        samplePositions1 = np.ceil(samplePositions1)
        samplePositions1.astype(np.int64)
        samplePositions2 = np.ceil(samplePositions2)
        samplePositions2.astype(np.int64)
        samplePositions = np.intersect1d(samplePositions1, samplePositions2)
        samplePositions = samplePositions.astype(np.int64, casting='unsafe', copy=True)
        print(samplePositions.dtype)
#        return samplePositions1, samplePositions2, samplePositions 
#        raw_input("Sis er ..")

        #Find the shared positions and take out the corr parts of the arrays:
#        firstPos = max(samplePositions1[0], samplePositions2[0] )
#        lastPos = max(samplePositions1[L1-1], samplePositions2[L2-1] )
#        firstIdx1 = np.where(samplePositions1 == firstPos)
#        lastIdx1 =  np.where(samplePositions1 == lastPos)
#        firstIdx2 = np.where(samplePositions2 == firstPos)
#        lastIdx2 =  np.where(samplePositions2 == lastPos)

#        firstPos = samplePositions[0]
#        L = samplePositions.shape[0]
#        lastPos = samplePositions[L-1]      
#        firstIdx1 = np.where(samplePositions1 == firstPos)
#        lastIdx1 =  np.where(samplePositions1 == lastPos)
#        firstIdx2 = np.where(samplePositions2 == firstPos)
#        lastIdx2 =  np.where(samplePositions2 == lastPos)

        #this gives the indices of the intersection elts in the original arrays 
        #(of which the intersection is made)
        idxs1 = np.searchsorted(samplePositions1, samplePositions)
        idxs2 = np.searchsorted(samplePositions2, samplePositions)

        predArray1_shared = predArray1.take(idxs1, axis = 0)
        qualArray1_shared = qualArray1.take(idxs1, axis = 0)
#        labelArray1_shared = labelArray1.take(idxs1)
        
        predArray2_shared = predArray2.take(idxs2, axis = 0)
        qualArray2_shared = qualArray2.take(idxs2, axis = 0)
        labelArray2_shared = labelArray2.take(idxs2, axis = 0)
        labelArray1_shared = labelArray2_shared
        
        if not( np.array_equal(labelArray1_shared, labelArray2_shared)):
            print("Warning: label arrays for the two models are diff!!")

        #Step 3:
        
        if LRtest_b == 1:

            #Predictions are now loaded. So run the LR-test (obs: samplePositions2 == samplePositions1 by construction):
            pVal, testFig, n, LR, var = loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, modelNameFrq = modelName1, modelNameNN = modelName2 , useSubSample_b = useSubSampleLR_b, subSampleSize = subSampleSizeLR, subSampleFraction = subSampleFractionLR, predArrayFrqModel = predArray1_shared, qualifiedArrayFrqModel = qualArray1_shared, k = flankSize1, predArrayNN = predArray2_shared, qualifiedArrayNN = qualArray2_shared, flankSize = flankSize2, dumpResults_b = 1, rootOutput = rootOutput_thisChromo)

#            #dump results:
#            dumpFile = rootOutput_thisChromo + 'generalizedLRtest_' + modelName1 + '_vs_' + modelName2 +  '_avgRevCompl1' + str(averageRevComplementary1_b) +  '_avgRevCompl2' + str(averageRevComplementary2_b) + '_' + chromoName
#            pickle.dump((pVal, testFig, n, LR, var), open(dumpFile, "wb") )
#
        if plot_b == 1:
            
            qualArray_shared = np.multiply(qualArray1_shared, qualArray2_shared)
            plotRefPredVsRefPred(labelArray = labelArray1_shared, predArray1 = predArray1_shared , predArray2 = predArray2_shared, qualArrayShared = qualArray_shared, modelName1 = modelName1, modelName2 = modelName2, bins = bins, rootOutput = rootOutput_thisChromo, chromoId = chromoName, annotationDict = annotationDict, positionsArray = samplePositions, startAtPosition = startAtPosition, nameChangeDict = nameChangeDict, saveAtDpi = saveAtDpi)


#Version of compareTwoModelsOnChromosomes (simple mod of) that compares one model to each model in 
#a list of models. To be used for comparing a NN model to eg a list of k-mer models.
#The NN-model predictions/arrays then only need to be loaded once (this is the most
#time consuming part of it all, it seems).
def compareOneModelToListOfModelsOnChromosomes(rootGenome, 
                                  chromosomeOrderList, 
                                  firstChromoNr,
                                  chromosomeDict, 
                                  rootOutput, 
                                  rootPredictModel1, 
                                  modelFile1, 
                                  flankSize1, 
                                  modelType1, 
                                  modelName1, 
                                  rootPredictModel2List = [], 
                                  modelFile2List = [], 
                                  flankSize2List = [], 
                                  modelType2List = [], 
                                  modelName2List = [],
                                  frqModelCoverType = 'All chromos', #else: 'Per chromo'
                                  averageRevComplementary2List_b = 0, 
                                  segmentLength2List = [1e6], 
                                  averageRevComplementary1_b = 0, 
                                  segmentLength1 = 1e6, 
                                  LRtest_b = 1, 
                                  useSubSampleLR_b =1, 
                                  subSampleSizeLR = 10000000,
                                  subSampleFractionLR = 0.1,
                                  plot_b = 1, 
                                  bins = 50, 
                                  chromoFieldIdx = 5,
                                  annotationTypes = [],
                                  rootAnnotationFiles = '',
                                  nameChangeDict = {},
                                  saveAtDpi = 100):
    
    '''Wrapper of LR tests and plotting of prediction arrays.

    modelType*: Indicates from what source the preictions are had.
                0: internal NN model; 1: internal frq model; 2: external model, Yuhu's folder notation!    

    frqModelCoverType: controls whether results from a frq model (k-mer model) should be 'Per chromo' or not (def: 'All chromos'). 

    '''   
    
    #Step 1: load prediction arrays forboth models and assemble them
    #Step 2: Check that samplePositions are consecutive and take "intersections" of all arrays.
    #Step 3: Compute LR test and plot as wanted
    
    
    #Step 1:

    #Depending on modelType generate dictionaries mapping each key in the chromoDict (chromo name) to the
    #path where to find the predictions
    predictDict1 = {}
    i = firstChromoNr + 1
    for chromoName in chromosomeOrderList:
        
        if modelType1 == 0:
            
            predictDict1[chromoName] = rootPredictModel1 + chromoName + '/'
            
        elif modelType1 == 1:
              
            if frqModelCoverType == 'Per chromo':  
                #typically like 'path/chr22/frqModel_chr22_k" + str(k) + ".txt"'
                chromoFileId = chromoName[chromoFieldIdx:] + '/frqModel_' + chromoName[chromoFieldIdx:] + '_k' + str(flankSize1) + ".txt" 
            else:
                #typically like 'path/frqModel_k" + str(k) + ".txt"'
                chromoFileId = '/frqModel_k' + str(flankSize1) + ".txt" 

            predictDict1[chromoName] = rootPredictModel1 + chromoFileId            
            
        elif modelType1 == 2:
            
            #typically like 'path/23.CHR22/Chr22.probs'
            if i > 9:
                chromoFileId = str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'
            else:
                chromoFileId = '0' + str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'

            i -= 1
            predictDict1[chromoName] = rootPredictModel1 + chromoFileId
            
#    print(predictDict1)
            
            
    #same for second model, though a list:
    predictDict2List = []
    #loop over model2-list:
    model2Cnt = 0
    for rootPredictModel2 in rootPredictModel2List:
        
        print("rootPredictModel2 ", rootPredictModel2)
        
        predictDict2 = {} #reset
        
        modelType2 = modelType2List[model2Cnt]
        flankSize2 = flankSize2List[model2Cnt]
        i = firstChromoNr + 1
        for chromoName in chromosomeOrderList:
            
            if modelType2 == 0:
                
                predictDict2[chromoName] = rootPredictModel2 + chromoName + '/'
                
            elif modelType2 == 1:
                          
                if frqModelCoverType == 'Per chromo':  
                    
                    #typically like 'path/chr22/frqModel_chr22_k" + str(k) + ".txt"'
                    chromoFileId = chromoName[chromoFieldIdx:] + '/frqModel_' + chromoName[chromoFieldIdx:] + '_k' + str(flankSize2) + ".txt" 
    
                else:
                    
                    #typically like 'path/frqModel_k" + str(k) + ".txt"'
                    chromoFileId = '/frqModel_k' + str(flankSize2) + ".txt" 
    
    
                predictDict2[chromoName] = rootPredictModel2 + chromoFileId            
     
               
            elif modelType2 == 2:
                
                #typically like 'path/23.CHR22/Chr22.probs'
                if i > 9:
                    chromoFileId = str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'
                else:
                    chromoFileId = '0' + str(i) + '.CHR' + str(i-1) + '/C' + chromoName[6:] + '.probs'

                i -= 1
                predictDict2[chromoName] = rootPredictModel2 + chromoFileId
    
        print("predictDict2 ", predictDict2)
        
        #record result:
        predictDict2List.append(predictDict2)
        model2Cnt += 1
    
#    raw_input("Sis, bare du ..")
    
    #Big outer loop over chromos; to be followed by inner loop over model2-list
    for chromoName in chromosomeOrderList:
        
        if not(chromoName in chromosomeDict):
            print("Chromo with name %s not found" % chromoName)
            continue
        
        fileName = chromoName + ".txt"
        fileGenome = rootGenome +fileName
                        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition, endAtPosition = chromosomeDict[chromoName]
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b =1
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

#        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString =  encodedGenomeData 

#        print(encodedGenomeData[3][:100])

        #Later we'll call assemblePredictArrayFromSegments, where we need to specif a nr-of-segments; this could eg
        #be set to some high number, eg 1000 but let us do it a little more carefully:
        L = len(encodedGenomeData[0]) #length of genome seq               
        nrSegments1 = int(np.ceil(float(L)/segmentLength1)) + 1 
        
#        raw_input("S..... er dejlig")

        #Create a directory if its not already there:
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):

            os.makedirs(rootOutput_thisChromo)
            print("Directory " + rootOutput_thisChromo + " created. Output will be placed there.")
    

        #load annotationArray if there:
        annotationDict = {}
        for annoType in annotationTypes:
            
            annotationFile = rootAnnotationFiles + chromoName + '_annotationArray_' + annoType
            
            print(annotationFile)
        
            
            try:
            
                annotationDict[annoType] = pickle.load(open(annotationFile,"rb"))
                
            except IOError:
                
                print("Annotation file %s not found" %  annotationFile)
                continue


    
        genomeIdName1 = chromoName #+ '_seg' + str(int(segmentLength1))    

        ####################################################
        
        #Now get the predictions:
        
        ####################################################
        
        #paths to where to find the predictions for model1:
        rootOutput1 = predictDict1[chromoName] 
 
        if modelType1 == 0:
            
            #To obtain a complete prediction array across all segments 
            #(use the same encodedData as when running the prediction across the genome (step1)!):                     
            predArray1, labelArray1, qualArray1, samplePositions1, samplePositionsBoolean1 = assemblePredictArrayFromSegments(rootOutput = rootOutput1, modelFileName = modelFile1, genomeIdName = genomeIdName1, nrSegments = nrSegments1, augmentWithRevComplementary_b = averageRevComplementary1_b, segmentLength = segmentLength1)                     
      
        elif modelType1 == 1:
         
            #temp placeholders
            samplePositions = 0
            samplePositionsIndicatorArray = 0            
            sampleGenomeSequence = ''
            
            resDictFrq = frqM.readResults(rootOutput1)
            getAllPositions_b = 1
            predArray1, qualArray1, samplePositions1, samplePositionsBoolean1, labelArray1 = getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, sampleGenomeSequence = sampleGenomeSequence, labelArray = '', fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel=resDictFrq, k= flankSize1)    
            #Then predArray1, qualArray1, samplePositions1, samplePositionsBoolean1 = predArrayFrq, qualArrayFrq, samplePositions, samplePositionsIndicatorArray 
        
        #tech necessesity
        samplePositions1 = np.ceil(samplePositions1) 
        samplePositions1 = samplePositions1.astype(np.int64, casting='unsafe', copy=True)         
        #Check that the samplePositions arrays are "without holes":
        L1 = samplePositions1.shape[0]
        checkDiff1 = samplePositions1[1:] - samplePositions1[:(L1-1)] #should be array of 1's
        check1 =  np.where(checkDiff1 > 1)
        if len(check1[0]) > 1:
            print("Obs! samplePositions1 has %d holes!" % len(check1[0]))
        else:
            print("samplePositions1 has no holes")
        
        
        #inner loop over model2-list:
        model2Cnt = 0
        for rootPredictModel2 in rootPredictModel2List:
            
            predictDict2 = predictDict2List[model2Cnt]
        
            modelType2 = modelType2List[model2Cnt]
            segmentLength2 = segmentLength2List[model2Cnt]
            modelFile2 = modelFile2List[model2Cnt]
            averageRevComplementary2_b = averageRevComplementary2List_b[model2Cnt]
            flankSize2 = flankSize2List[model2Cnt]
            modelName2 = modelName2List[model2Cnt]
            
            nrSegments2 = int(np.ceil(float(L)/segmentLength2)) + 1 
            
            genomeIdName2 = chromoName + '_seg' + str(int(segmentLength2)) 
            rootOutput2 = predictDict2[chromoName]
            
            print(rootOutput2)
    
            if modelType2 == 0:
    
                predArray2, labelArray2, qualArray2, samplePositions2, samplePositionBoolean2 = assemblePredictArrayFromSegments(rootOutput = rootOutput2, modelFileName = modelFile2, genomeIdName = genomeIdName2, nrSegments = nrSegments2, augmentWithRevComplementary_b = averageRevComplementary2_b, segmentLength = segmentLength2)                     

     
            elif modelType2 == 1 or modelType2 == 2:
                
                #For the frq model/external (model2) use the sampledPositions from the NN model (model1):
                getAllPositions_b = 0
                samplePositions = samplePositions1
                labelArray = labelArray1
                samplePositionsIndicatorArray = samplePositionsBoolean1
                sampleGenomeSequence = encodedGenomeData[3] # genomeSeqString is 3rd entry of encodedGenomeData
         
    
                if modelType2 == 1:
                     
                    resDictFrq = frqM.readResults(rootOutput2)
                    predArray2, qualArray2, samplePositions2, samplePositionsBoolean2, labelArray2 = getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, samplePositionsIndicatorArray = samplePositionsIndicatorArray, sampleGenomeSequence = sampleGenomeSequence, labelArray =labelArray, fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition=startAtPosition, endAtPosition=endAtPosition, resultsDictFrqModel=resDictFrq, k= flankSize2)    
                    #Then: predArray2, qualArray2, samplePositions2, sampledPositionsBoolean2 = predArrayFrq, qualArrayFrq, samplePositions, samplePositionsIndicatorArray
    
                elif modelType2 == 2:
                    
                    #Example is the Markov model. There we read in the prob's from Yuhu's run of AK's model. 
                    #Read in prob's from external: 
                    positionArrayMarkov, predArrayMarkov, refBaseListMarkov = readInProbsFromExternal(rootOutput2)
                    #We have to "standardize" these so that the arrays match those from the other model's indexing (here a NN-model).
                    displacementOfExternal = 1 #indexing start from 1 in results from Markov model
                    predArray2, qualArray2, samplePositions2, labelArray2 = getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = samplePositions, sampleGenomeSequence = sampleGenomeSequence, labelArray =labelArray, positionsExternal= positionArrayMarkov, probsExternal= predArrayMarkov, refBaseListExternal = refBaseListMarkov,  fileGenome =fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= flankSize2, displacementOfExternal = displacementOfExternal)
                    #Then predArray2, qualArray2, samplePositions2 = predArrayMarkov_shared, qualArrayMarkov_shared, samplePositionsMarkov_shared  
                    
    
    
            #Step 2:
            
            #Check that the samplePositions arrays are "without holes" (done above for model1):
            L2 = samplePositions2.shape[0]
            checkDiff2 = samplePositions2[1:] - samplePositions2[:(L2-1)] #should be array of 1's
            check2 =  np.where(checkDiff2 > 1)
            if len(check2[0]) > 1:
                print("Obs! samplePositions2 has %d holes!" % len(check2[0]))
            else:
                print("samplePositions2 has no holes")
    
                
            #samplePositions1: see above (outside model2-list loop)
            samplePositions2 = np.ceil(samplePositions2)
            samplePositions2.astype(np.int64)
            samplePositions = np.intersect1d(samplePositions1, samplePositions2)
            samplePositions = samplePositions.astype(np.int64, casting='unsafe', copy=True)
            print(samplePositions.dtype)
    #        return samplePositions1, samplePositions2, samplePositions 
    #        raw_input("Sis er ..")
        
            #this gives the indices of the intersection elts in the original arrays 
            #(of which the intersection is made)
            idxs1 = np.searchsorted(samplePositions1, samplePositions)
            idxs2 = np.searchsorted(samplePositions2, samplePositions)
    
            predArray1_shared = predArray1.take(idxs1, axis = 0)
            qualArray1_shared = qualArray1.take(idxs1, axis = 0)
    #        labelArray1_shared = labelArray1.take(idxs1)
            
            predArray2_shared = predArray2.take(idxs2, axis = 0)
            qualArray2_shared = qualArray2.take(idxs2, axis = 0)
            labelArray2_shared = labelArray2.take(idxs2, axis = 0)
#4 lines following are out-commented since labelArray1_shared is no used; the check that 
#the two are id -- if labelArray1_shared was properly defined -- is not needed since the
#check is handled in the getPredArrayFromExternal and getPredArrayFrqModel fct's:
#            labelArray1_shared = labelArray2_shared
                        
#            if not( np.array_equal(labelArray1_shared, labelArray2_shared)):
#                print("Warning: label arrays for the two models are diff!!")
    
            #Step 3:
            
            if LRtest_b == 1:
    
                #Predictions are now loaded. So run the LR-test (obs: samplePositions2 == samplePositions1 by construction):
                pVal, testFig, n, LR, var = loglikelihoodRatioTestNonNestedModels(encodedGenomeData = encodedGenomeData, samplePositions = samplePositions, modelNameFrq = modelName2, modelNameNN = modelName1 , useSubSample_b = useSubSampleLR_b, subSampleSize = subSampleSizeLR, subSampleFraction = subSampleFractionLR, predArrayFrqModel = predArray2_shared, qualifiedArrayFrqModel = qualArray2_shared, k = flankSize2, predArrayNN = predArray1_shared, qualifiedArrayNN = qualArray1_shared, flankSize = flankSize1, dumpResults_b = 1, rootOutput = rootOutput_thisChromo)
    
    #            #dump results:
    #            dumpFile = rootOutput_thisChromo + 'generalizedLRtest_' + modelName1 + '_vs_' + modelName2 +  '_avgRevCompl1' + str(averageRevComplementary1_b) +  '_avgRevCompl2' + str(averageRevComplementary2_b) + '_' + chromoName
    #            pickle.dump((pVal, testFig, n, LR, var), open(dumpFile, "wb") )
    #
            if plot_b == 1:
                
                qualArray_shared = np.multiply(qualArray1_shared, qualArray2_shared)
                plotRefPredVsRefPred(labelArray = labelArray2_shared, predArray1 = predArray2_shared , predArray2 = predArray1_shared, qualArrayShared = qualArray_shared, modelName1 = modelName2, modelName2 = modelName1, bins = bins, rootOutput = rootOutput_thisChromo, chromoId = chromoName, annotationDict = annotationDict, positionsArray = samplePositions, startAtPosition = startAtPosition, nameChangeDict = nameChangeDict, saveAtDpi = saveAtDpi)

            model2Cnt += 1


 
def plotMaxPredVsMaxPred(predArray1 , predArray2, qualArrayShared, modelName1, modelName2, bins = 50, rootOutput = '', saveAtDpi = 100):
    '''Make a scatter plot of the two max-predictions against each other: each
    position of the arrays consists in a distribution (of the four bases); taking
    the max we get the highest probability at each position. These are then 
    shown in a binned 2d scatter plot.
    This amounts to interpreting the max-output from each model as a reference ("the reference defined by the
    model").'''
    
    #Check that all arrays have same shape:
    if qualArrayShared.shape[0] != predArray1.shape[0] or qualArrayShared.shape[0] != predArray2.shape[0]:
        
        print("Input arrays have different shapes -- so exits here!")
        return

    
    #build the arrays of max-predictions only for the positions having qualArrayShared[pos] = 1   
    L = np.sum(qualArrayShared)
    #maxPredArray1 = np.zeros(shape = L)
    #maxPredArray2 = np.zeros(shape = L)
    
    idxs = np.where(qualArrayShared > 0)[0]
    maxPredArray1 = predArray1.take(idxs, axis = 0)
    maxPredArray2 = predArray2.take(idxs, axis = 0)
    
    maxPredArray1 = maxPredArray1.max(axis = 1)
    #maxPredArray1_ = maxPredArray1[1:] 
    maxPredArray2 = maxPredArray2.max(axis = 1)
    #maxPredArray2_ = maxPredArray2[:(L-1)]
#     
#    cnt = 0
#    for i in range(qualArrayShared.shape[0]):
#        
#        if qualArrayShared[i] == 0:  
#            
#            continue
#
#        else:
#            
#            maxPredArray1[cnt] = predArray1[i].max(axis = 0)
#            maxPredArray2[cnt] = predArray2[i].max(axis = 0)
#            cnt +=1
#            
#    print("Processed %d of %d", cnt, L )
        
    
    
        
    plt.figure()
    plt.title('Max-predictions of two models' )
    hist2dCounts = plt.hist2d(maxPredArray1, maxPredArray2, bins =bins, norm=LogNorm()) 
    plt.xlabel(modelName1)   
    plt.ylabel(modelName2)    
    plt.colorbar()
    plt.show()
    plt.savefig(rootOutput + modelName1 + '_vs_' + modelName2 + '_maxPredictions_hist2dPlot.pdf', dpi = saveAtDpi) 
     
        



def plotRefPredVsRefPred(labelArray, predArray1 , predArray2, qualArrayShared, modelName1, modelName2, bins = 50, rootOutput = '', chromoId = '', annotationDict = {}, positionsArray = 0, nameChangeDict = {}, modelNameChangeDict= modelNameChangeDict, startAtPosition = 0, fontSize = 'large', saveAtDpi = 100):
    '''Make a scatter plot of the two predictions against each other: each
    position of the arrays consists in a distribution (of the four bases); taking at each position 
    the one of these that corr's to the true base (ref-base) we get the model's probability of predicting the ref-base
    at each position. These are then shown in a binned 2d scatter plot.
    
    labelArray: one-hot encoded genome seq (as output by dataGen.encodeGenome). MUST corr to prediction arrays.
    predArray1 , predArray2: prediction arrays from the two models (refering to exactly the same genomic sequence)
    qualArrayShared: array of 0/1 indicator revealing the positions at which both models produced a prediction 
    
    chromoId: just an optional char that will be added to the file name upon write-out. 
    '''
    
    #Check that all arrays have same shape:
    if qualArrayShared.shape[0] != predArray1.shape[0] or qualArrayShared.shape[0] != predArray2.shape[0] or labelArray.shape[0] != qualArrayShared.shape[0]:
        
        print("Input arrays have different shapes -- so exits here!")
        return

    
    #build the arrays of predictions of the reference only for the positions having qualArrayShared[pos] = 1      
    idxs = np.where(qualArrayShared == 1)[0] #changed from:np.where(qualArrayShared > 0)[0]
    qLabelArray = labelArray.take(idxs, axis = 0) #[1:]
    qPredArray1 = predArray1.take(idxs, axis = 0) #[1:]
    qPredArray2 = predArray2.take(idxs, axis = 0) #[:(len(idxs) -1)]
    
    probArray1 = np.zeros(shape = len(idxs)) #changed from np.zeros(shape = len(idxs)-1)
    probArray2 = np.zeros(shape = len(idxs)) #ditto
    #Compute the dot products:
    for i in range(len(idxs)): #changed from len(idxs)-1
        
        if max(qLabelArray[i]) > 1:
            continue
        
        probArray1[i] = np.dot(qPredArray1[i], qLabelArray[i])   
        probArray2[i] = np.dot(qPredArray2[i], qLabelArray[i])

           
    fig = plt.figure()
    plt.title('All positions', fontsize = fontSize)
    hist2dCounts, x_bins, y_bins, img = plt.hist2d(probArray1, probArray2, bins = bins, norm=LogNorm(1, len(idxs) )) 
   
    if modelName1 in modelNameChangeDict:
        labelModelName1 = modelNameChangeDict[modelName1]
    else:    
        labelModelName1 = modelName1
    if modelName2 in modelNameChangeDict:
        labelModelName2 = modelNameChangeDict[modelName2]
    else:    
        labelModelName2 = modelName2
        
    plt.xlabel(labelModelName1, fontsize = fontSize) 
    plt.ylabel(labelModelName2, fontsize = fontSize)    
    plt.colorbar()
    plt.show()
    
    print("Sum counts in input and histo: ", len(idxs), np.sum(hist2dCounts))
#    #dump the fig data:
#    dumpFile = rootOutput + modelName1 + '_vs_' + modelName2 + '_refPredictions_hist2dPlot_figData_' + chromoId + '.pkl'
#    pickle.dump(fig, open(dumpFile, 'wb'))
    
    plt.savefig(rootOutput + modelName1 + '_vs_' + modelName2 + '_refPredictions_hist2dPlot_' + chromoId +  '.pdf', dpi = saveAtDpi) 
    plt.close()

    

#    cmap = cm.get_cmap('Set1')
#    figScatter = plt.figure()
#    plt.title('Annotated prediction of reference acc to two models' )
#    plt.scatter(probArray1, probArray2, c = cmap(0), alpha = 0.5)
    colorNr = 1
    for annoType in annotationDict:
        
        annoName = annoType
        #change name if wanted
        if annoType in nameChangeDict:
            
            annoName = nameChangeDict[annoType]
        
        fig = plt.figure()
        plt.title(annoName , fontsize = fontSize)
        
        print("Doing scatter plot for annotype %s" %  annoType)
         
        annoArray =  annotationDict[annoType]
        whereThisAnno = np.where(annoArray > 0.5)[0] # >0 should do, but just to be on the safe side
        qPositionsArray = positionsArray.take(idxs, axis = 0) #positions having qual == 1
        sharedElements = np.intersect1d(whereThisAnno, qPositionsArray + startAtPosition)
        #find indices of the shared idxs in the positionsArray :
        relevantIdxs = np.searchsorted(qPositionsArray + startAtPosition, sharedElements)
        
#        #The qualified of these are then (idxs: where qual == 1, see above):
#        relevantIdxsQual = np.intersect1d(relevantIdxs, idxs) 
        
#        print(annoType, whereThisAnno.shape, positionsArray.shape)
        
        #The annoArray is then an array of 0/1's indicating the annoType covering exactly the positions sampled
        #for the predictions. To get the prob'arrays for the annoType is then simple:
        #just "take" them in the prob-arrays:
        
        probArray1_thisAnno = probArray1.take(relevantIdxs)
        probArray2_thisAnno = probArray2.take(relevantIdxs)
        
#        axScatter.scatter(probArray1_thisAnno, probArray2_thisAnno, label = annoType, c = cmap(colorNr), alpha = 0.5)
        ret = plt.hist2d(probArray1_thisAnno, probArray2_thisAnno, bins = (x_bins, y_bins), norm = LogNorm())
        plt.xlabel(labelModelName1, fontsize = fontSize) 
        plt.ylabel(labelModelName2, fontsize = fontSize)    
        plt.colorbar()

#        #dump the fig data:
#        dumpFile = rootOutput + modelName1 + '_vs_' + modelName2 + '_refPredictions_annotated_hist2dPlot_figData_' + chromoId  + '.pkl'
#        pickle.dump(fig, open(dumpFile, 'wb'))
       
        plt.savefig(rootOutput + modelName1 + '_vs_' + modelName2 + '_refPredictions_annotated_hist2dPlot_' + annoType + '_' + chromoId + '.pdf' , dpi = saveAtDpi) 
        
       
        plt.close()

        colorNr += 1
    
#    axScatter.legend()

#    axScatter.savefig(rootOutput + modelName1 + '_vs_' + modelName2 + '_refPredictions_annotatedScatterPlot.pdf' ) 
    
#    axScatter.close()
        



def autoCorrAvgPred(avgPredArray, windowSize, stepSize, qualArray, maxHorizon = 1000):
    
    '''Computes the autocorrelation of the input prediction array (as output by predictOnGenomeSamples and getPredArrayFrq)
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
    refBaseList = []
    
    with open(fileName, mode='r') as probsFile:
        
        probsReader = csv.reader(probsFile, delimiter='\t')
        
        lineCnt =  0
        for row in probsReader:
            
            if lineCnt < 2:
                lineCnt +=1
                continue
            
            #Record the refernece base:
            refBase = row[3]
            refBaseList.append(refBase)
            
            
            E_P = row[7].split(';') #splits the info in a list with two entries: 'E= ..." and "P=..."            
            PstringRaw = E_P[1] 
            #fetch the part of the Pstring after "P=":
            Pstring = PstringRaw.partition('P=')[2]
            #split the Pstring
            PstringSplit = Pstring.split(',')
#            print(PstringSplit)
            Pvalues = map(float, PstringSplit)
            
            posList.append(int(row[1]))
            probsList.append(Pvalues)                        
            
            lineCnt +=1
        
        
    return np.asarray(posList), np.asarray(probsList), refBaseList
    
    
    
def getPredArrayFromExternal(getAllPositions_b, samplePositions, sampleGenomeSequence, labelArray, positionsExternal, probsExternal, refBaseListExternal, fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, k, displacementOfExternal = 1):

    '''
    Similar to getPredArrayFrqModel. Here though the prob's of the four bases at every position are written
    in from an external source (by readInProbsFromExternal). And we fetch these prob's at a desired set of samplePositions.
    
    Input:
        
    sampleGenomeSequence: the (full) genomic letter-seq from which the sampled positions, at samplePositions, are taken (ie at samplePositions[i] we have the letter/base sampleGenomeSequence[samplePositions[i]]).
    samplePositions: set of positions RELATIVE TO startAtPosition at which we want to have the predictions (usually had from getting the predictions from an internal model, ie a NN model)
    getAllPositions_b: if 1 all positions will be sampled ###old rubbish?: the samplePositionIndicatorArray will consist of 1's for the length of the read-in genomic data (which is done by fastReadGenome)
    labelArray: encoded corr to the input sampleGenomeSequence
    displacementOfExternal: the index at which the external positions are counted from (where the indexing starts from --- in practice 0 or 1). Default = 1.
    
    Returns: 
    
    predArray, qualArray, samplePositions, sampleGenomeSequenceEncoded
    
    where: 
    predArray: the prediction of the Markov model at the position given by the elt of samplePositions at the same index (in the array)
    qualArray: boolean to reveal whether or not the position was covered by the Markov model
    samplePositions: the input sample positions (which are RELATIVE to the startAtPosition)
    sampleGenomeSequenceEncoded: the one-hot encoded genome sequence corr to the sample positions
 
    '''
    
    if getAllPositions_b == 1:
        
        #Read the genome seq (X is the string in upper case):
        Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileName = fileGenome, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = 0,
#               outputGenomeString_b = 0,
               randomChromo_b = 0, 
               avoidChromo = [])
    
        lenX = len(X)

#        lenPred = lenX
#        nrSamples =  lenX - 2*k - 1
        samplePositions = np.arange(k, lenX - k - 1) #, dtype='int64')
        samplePositionsAbs = samplePositions + startAtPosition 
        
        sampleGenomeSequence = X #So: the samplePositions array give the indices of this seq that are sampled; eg the 0 entry of samplePositions is k, corr to the k'th letter of X.
        
#        for i in range(lenX):
#            samplePositions[i] = i
            
        lenPred = samplePositions.shape[0] #len(positionsExternal) #
        
    else:
        
        print("getAllPositions_b is set to 0, so an array of samplePositions and corresponding genomeSequence must be provided")
    
        lenPred = samplePositions.shape[0]  #= number of samples (some may be repeated)
        
        #the input sample positions are relative to startAtPosition:
        samplePositionsAbs = samplePositions + startAtPosition

    
    predArray = np.zeros(shape=(lenPred, 4)) #data type is float64 by default
    Q = np.zeros(shape = lenPred, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 

    #We want to include in the output the one-hot encoded genome seq corr to the samplePositions (ie the label array):
    sampleGenomeSequenceEncoded = np.zeros(shape = (lenPred, 4), dtype = 'int8')   
   
    #Loop through the genome seq, get the prob's at each position covered.
#    #It seems easiest to consider the intersection and its complement as follows:
#    sharedPositions = np.intersect1d(samplePositionsAbs, positionsExternal)
#    notSharedPositions = np.setdiff1d(samplePositionsAbs, positionsExternal) 
    
    #Get indices of the externally "sampled" positions in the samplePositionsAbs;
    #we use searchsorted, so see to that both arrays are sorted in ascending order:
    extIdxs_left = np.searchsorted(samplePositionsAbs, positionsExternal - displacementOfExternal, 'left')
    extIdxs_right = np.searchsorted(samplePositionsAbs, positionsExternal - displacementOfExternal, 'right')
    #The positions for which the left and right-based idx's are different (will diff by 1)
    #are the ones we want (:corr to all elts of positionsExternal in samplePositionsAbs )
    cntId = 0
    for i in range(len(positionsExternal)):

        l = extIdxs_left[i]        
        r = extIdxs_right[i]
        
        if l == r:
            
            if l <= 0 or l >= lenPred:
                continue
            else: 
               #put in a random bid:
               randomIdx = np.random.randint(0,4) 
               predArray[l][randomIdx] = 1.0   
               #Q[l] is 0 by def
               
        elif r == l+1:

            cntId += 1             
            #Check if the reference bases are the same:
            pos = samplePositions[l]
            if sampleGenomeSequence[pos] == refBaseListExternal[i]:
                
                predArray[l] = probsExternal[i]
                Q[l] = 1  
                
                #for the corr array of labels:
                #fastReadGenome returns bases in upper-case: 
                if sampleGenomeSequence[pos] == 'A':            
                    sampleGenomeSequenceEncoded[l] = dataGen.codeA_asArray
                elif sampleGenomeSequence[pos] == 'T':     
                    sampleGenomeSequenceEncoded[l] = dataGen.codeT_asArray
                elif sampleGenomeSequence[pos] == 'C':    
                    sampleGenomeSequenceEncoded[l] = dataGen.codeC_asArray
                elif sampleGenomeSequence[pos] == 'G':    
                    sampleGenomeSequenceEncoded[l] = dataGen.codeG_asArray
                else:
                    sampleGenomeSequenceEncoded[l] = dataGen.codeW_asArray #wild-card array 
            
                
        
        else:
            #put in a random bid:
           randomIdx = np.random.randint(0,4) 
           predArray[l][randomIdx] = 1.0
           #Q[l] is 0 by def
            
        

#    for idx in samplePositionsAbs:
#        
#        #find index of idx in positionsExternal;if doesn't exist skip:
#        extIdx = np.where(positionsExternal == idx)[0]
#        if not extIdx: #not(np.isin(idx, positionsExternal)):        
#
#            Q[i] = 0
#             #put in a random bid:
#            randomIdx = np.random.randint(0,4) 
#            predArray[i][randomIdx] = 1.0
#            
#        else:
#
#            print "extIdx ", extIdx[0]
#            predArray[i] = probsExternal[extIdx[0]]
#        

#        i += 1


    print("Nr of id ref's: ", cntId )
    print("Nr of qualified: ", np.sum(Q))

#    #We want to include in the output the one-hot encoded genome seq corr to the samplePositions (ie the label array):
#    sampleGenomeSequenceEncoded = np.zeros(shape = (lenPred, 4), dtype = 'int8')
#    for i in range(lenPred):
#        
#        pos = samplePositions[i]
#        #fastReadGenome returns bases in upper-case: 
#        if sampleGenomeSequence[pos] == 'A':            
#            sampleGenomeSequenceEncoded[i] = dataGen.codeA_asArray
#        elif sampleGenomeSequence[pos] == 'T':     
#            sampleGenomeSequenceEncoded[i] = dataGen.codeT_asArray
#        elif sampleGenomeSequence[pos] == 'C':    
#            sampleGenomeSequenceEncoded[i] = dataGen.codeC_asArray
#        elif sampleGenomeSequence[pos] == 'G':    
#            sampleGenomeSequenceEncoded[i] = dataGen.codeG_asArray
#        else:
#            sampleGenomeSequenceEncoded[i] = dataGen.codeW_asArray #wild-card array 
            
    #In case we have set getAllPositions_b = 0 and so have derived/obtained the 
    #predArray for the provided sampleGenomeSequence, the provided labelArray
    #should be id to the sampleGenomeSequenceEncoded that we have just derived:
    if getAllPositions_b == 0:
        
        if np.array_equal(sampleGenomeSequenceEncoded, labelArray):
            
            print("Fine, the derived label array corr's to the provided labelArray.")
        
        else:
            
            print("Warning: the derived label array is NOT id to the provided labelArray!")
            
            
            if sampleGenomeSequenceEncoded.shape !=  labelArray.shape:
                print("The two arrays have diff shapes: ", sampleGenomeSequenceEncoded.shape,  labelArray.shape)

            else:
                
                print("The two arrays have id shapes. We check if, where they differ, the position is disqualified:")
                cntTrueDiffs = 0
                for j in range(labelArray.shape[0]):
                    
                    if not(np.array_equal(labelArray[j], sampleGenomeSequenceEncoded[j])) and Q[j] == 1:
                        cntTrueDiffs += 1
                        
                if cntTrueDiffs == 0:
                    
                    print("No need to worry: there are no diff's on qualified positions")
                
                else:
                    
                    print("You need to worry: there are %d diff's on qualified positions" % cntTrueDiffs)
                    
                
                
#            print(sampleGenomeSequenceEncoded[100000:100010])
#            print(labelArray[100000:100010])

    #Obs: sampleGenomeSequenceEncoded is (id to) the label array
    return predArray, Q, samplePositions, sampleGenomeSequenceEncoded


def splitExternalPredArrayInSegments(rootOutput,
                                     modelFileName,
                                     averageRevComplementary_b,
                                     genomeIdName,
                                     predArray, 
                                     qualifiedArray,
                                     labelArray, 
                                     sampledPositionsArray,
                                     segmentLength):
    '''For splitting eg prediction array had from the Markov model in segments for the purpose
    of running the same analyses on that model as can be run on the NN models (eg Fourier transf's)'''
    
    L =  predArray.shape[0]
    nrSegments = int(np.floor(float(L)/segmentLength))   
    
    print("nrSegments: ", nrSegments)
#    raw_input("S er den ........")

    segStart = 0
    segEnd = segmentLength
    for j in range(nrSegments):
        
        print("Now at segment ", j)
        
        genomeIdNameSeg = genomeIdName + '_segment' + str(j)
        
        segStart = j*segmentLength
        segEnd = segStart + segmentLength
        
        predArraySeg = predArray[segStart:segEnd]
        qualifiedArraySeg = qualifiedArray[segStart:segEnd]
        sampledPositionsArraySeg = sampledPositionsArray[segStart:segEnd]
        labelArraySeg = labelArray[segStart:segEnd]
        
        #Keep a copy of the results:
        dumpFile = rootOutput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        pickle.dump(labelArraySeg, open(dumpFile, "wb") )
        dumpFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        pickle.dump(predArraySeg, open(dumpFile, "wb") )
        dumpFile = rootOutput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        pickle.dump(qualifiedArraySeg, open(dumpFile, "wb") )
        dumpFile = rootOutput + modelFileName + '_' + 'sampledPositions' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
        pickle.dump(sampledPositionsArraySeg, open(dumpFile, "wb") )
        
        



def externalModelPredictOnChromosomes(rootGenome, 
                         chromosomeDict,
                         chromosomeOrderList, 
                         rootOutput, 
                         averageRevComplementary_b,
                         kModel,
                         modelFileName,
                         modelPredicitionFileNameDict = {},
                         displacementOfExternal = 0, 
                         kMer_b = 0, 
                        segmentLength = 1e6,
                        startAtSegmentDict = {}, 
                        windowLength = 1,
                        stepSize = 1,
                        Fourier_b = 0,
                        on_binf_b = 1
                        ):
    '''Aim: to fetch the predictions of an external model (such as the Markov model) on each of a 
    list of chromosomes, and split it in segments (incl qual arrays etc as for the NN  models).
    
    chromosomeDict: key: chormosome name, values: [startAtPosition, endAtPosition]
    
    #For human DNA, assembly hg38 (start = startAtPosition):
    chr13: start 16000000
    chr14: start 16000000
    chr15: start 17000000
    chr21: start 5010000
    chr22: start 1050000 
    
    chromosomeOrderList: gives the order in which the function processes the 
    chromosomes. The list should only contain chromo names that are keys in chromosomeDict.
    
    kMer_b: if 0 read in prob's from file (Markov model); if 1 the prob's from kmer model (k = kModel)
    will be read in.
    
    modelPredicitionFileNameDict: if not k-mer model (kMer_b = 0) let this be a dict mapping each chromoname
    to be covered to a path to the file holding the the model predictions (prob's); else just leave it blank.
        
    '''
    
    #Loop over chromosomes
    # -- step1 read the prob's from file
    # -- step2 derive the pred-array
    # -- step3 split in segments
    
    
    for chromoName in chromosomeOrderList:
        
        #step1
        if kMer_b == 0:
            
            fileNameModelPrediction = modelPredicitionFileNameDict[chromoName] #path to file containing the prediction for the model on the given chromosome            
            positionArray, predArray, refBaseList = readInProbsFromExternal(fileNameModelPrediction)
        
        elif kMer_b == 1:
            
            fileName = rootOutput + "frqModel_k" + str(kModel) + ".txt"
            resultsDictFrqModel = frqM.readResults(fileName)
        
        
        #step2
        genomeFileName = chromoName + ".txt"
        fileGenome = rootGenome + genomeFileName #path to the genome/chromosome sequence
        
        getAllPositions_b = 1
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition, endAtPosition  = chromosomeDict[chromoName]

        if kMer_b == 0:
        
            predArray, qualArray, samplePositions, sampleGenomeSequenceEncoded = getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = 0, sampleGenomeSequence = '', labelArray = '', positionsExternal= positionArray, probsExternal= predArray, refBaseListExternal = refBaseList,  fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= kModel, displacementOfExternal = displacementOfExternal)
#                                                                                    getPredArrayFromExternal(getAllPositions_b, samplePositions, sampleGenomeSequence, labelArray, positionsExternal, probsExternal, refBaseListExternal, fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, k, displacementOfExternal = 1)

        elif kMer_b == 1:
        
            print("S er den .........")
            predArray, qualArray, samplePositions, samplePositionsIndicatorArray, sampleGenomeSequenceEncoded = getPredArrayFrqModel(getAllPositions_b = getAllPositions_b, samplePositions = 0,  sampleGenomeSequence ='', labelArray = '', fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition, resultsDictFrqModel = resultsDictFrqModel, k = kModel, samplePositionsIndicatorArray = 0)
            print(predArray.shape)
         
        #step3            
        genomeIdNameSeg = chromoName + '_seg' + str(int(segmentLength))
        

        #Create a directory if it's not already there:
        rootOutput_thisChromo = rootOutput +  chromoName + '/'
        if not os.path.exists(rootOutput_thisChromo):

            os.makedirs(rootOutput_thisChromo)
            print("Directory " + rootOutput_thisChromo + " created. Output will be placed there.")
        
        splitExternalPredArrayInSegments(rootOutput = rootOutput_thisChromo,
                                     modelFileName = modelFileName,
                                     averageRevComplementary_b = averageRevComplementary_b,
                                     genomeIdName = genomeIdNameSeg,
                                     predArray = predArray, 
                                     qualifiedArray = qualArray,
                                     labelArray = sampleGenomeSequenceEncoded, 
                                     sampledPositionsArray = samplePositions,
                                     segmentLength = segmentLength)
                                   



#For generating an array of 'random snps' with the same structure as the list output by snpAnalysis.fetchProbsAtSnps: a pair, (p1,p2), of probs for each of a randomly picked
#positions in a given chromosome, with p1= the prob of the reference base at the given position and p2 = the prob of one of the non-ref bases at the position (
#randomly picked, where prob's are acc to the given model
def makeRandomProbRefProbAltArray(pctPositions,
                                  assemblePredictArrayFromSegmentsOutput,
                                  rootOutput = '', 
                        modelFileName = '',
                        segmentLength = '',
                        genomeIdName = '', #for saving the prediction array; could eg be a chromo name
                        nrSegments = 0, #encodedGenomeData,
                        augmentWithRevComplementary_b = 0,
#                        annotationDict = {},
                        startAtPosition = 0,
                        repeatComplement_b = 0,
                        repeatAnnoTypes = [],
                        windowLength = 1,
                        stepSize = 1):
    ''' 
    'Random sibling' to fetchProbsAtSnps.  
    
    pctPositions: the fraction of all qualified positions to be randomly sampled
    
    assemblePredictArrayFromSegmentsOutput: 'assembled' predArray etc as output by the fct assemblePredictArrayFromSegments
    
    annotationDict: mapping annotation type to 0/1 indicator array of the annotation 
    
    Returns: dict mapping each annotation type to the array of corr pairs of prob's. Key 'all'
    provides the array for all sampled positions. 
    '''
    
#    resultsDict = {}
#    resultsDict['all'] = []
#    for annoType in annotationDict:
#
#        resultsDict[annoType] = []    

    #fetch assembly of the model's probs etc if nor provided:
    if len(assemblePredictArrayFromSegmentsOutput) != 5:
        predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = assemblePredictArrayFromSegments(rootOutput = rootOutput, 
                                modelFileName = modelFileName,
                                segmentLength = segmentLength,
                                genomeIdName = genomeIdName, #for saving the prediction array; could eg be a chromo name
                                nrSegments= nrSegments, #encodedGenomeData,
                                augmentWithRevComplementary_b = augmentWithRevComplementary_b,
                                on_binf_b = 1,
                                for_GCbias_b = 0,
                                rootGCbias = '', 
                                windowLength = windowLength,
                                stepSize = stepSize)
    else:
        predArray, labelArray, qualArray, sampledPositions, sampledPositionsBoolean = assemblePredictArrayFromSegmentsOutput

#    #Get annotation array corr to sampled positions. These must be rel to full chromo seq
#    sampledPositionsAbs = sampledPositions + startAtPosition
#    sampledPositionsAbs = sampledPositionsAbs.astype(np.int64, casting='unsafe', copy=True)
#    annoArrayDict = {}
#    for annoType in annotationDict:
#        
#        annoArrayDict[annoType] = annotationDict[annoType].take(sampledPositionsAbs)
#        
#        print("Shape of anno-array for anno ", annoType, annoArrayDict[annoType].shape)

#    #If desired switch to opposite booleans for the given repeatAnnoTypes
#    if repeatComplement_b == 1:
#        for annoType in repeatAnnoTypes:
#            annoArrayDict[annoType] = 1 - annoArrayDict[annoType]
        
        
    #Loop over the postions and pick the positions etc:
    ACGT = (dataGen.codeA_asArray, dataGen.codeC_asArray, dataGen.codeG_asArray, dataGen.codeT_asArray)
    L = predArray.shape[0]
    probsList = []
    for i in range(L):

        if qualArray[i] == 0:
            continue
        
        ind = np.random.binomial(1,pctPositions)
        if ind == 1:
            
            #prob of ref base:
            refProb = np.dot(predArray[i],labelArray[i])
                                       
            #pick an 'alternative base' and its prob:
            j = np.argmax(labelArray[i])
            cands = range(4)
            cands.pop(j)
            idxAlt = np.random.choice(cands, p = predArray[i].take(cands)/np.sum(predArray[i].take(cands))) #divide: to obtain prob distr -- really just the distr conditional on ref)
            altProb = np.dot(predArray[i],ACGT[idxAlt])
            
            triple = [i,refProb,altProb]
            probsList.append(triple)

    probsList = np.asarray(probsList)
    
    return probsList            
            
#            resultsDict['all'].append(triple)
#            
#            #and for the annos:
#            for annoType in annotationDict:
#
#                if annoArrayDict[annoType][i] == 1: 
#                    resultsDict[annoType].append(triple) 
#    
#    #tr to arrays:        
#    for annoType in resultsDict:
#        
#        resultsDict[annoType] = np.asarray(resultsDict[annoType])
#    
#    return resultsDict   


                                
    
#def frqModelPredictOnChromosomes(rootGenome, 
#                         chromosomeDict,
#                         chromosomeOrderList, 
#                         modelFileName, 
#                         modelPredicitionFileNameDict,
#                         averageRevComplementary_b,
#                         rootOutput,
#                         displacementOfExternal, 
#                         kModel,
#                         kMer_b = 0,
#                        segmentLength = 1e6,
#                        startAtSegmentDict = {}, 
#                        windowLength = 1,
#                        stepSize = 1,
#                        Fourier_b = 0,
#                        on_binf_b = 1
#                        ):
#    '''Aim: to fetch the predictions of a k-mer model on each of a list of chromosomes, and 
#    split it in segments (incl qual arrays etc as for the NN  models).
#    
#    chromosomeDict: key: chormosome name, values: [startAtPosition, endAtPosition]
#    
#    #For human DNA, assembly hg38 (start = startAtPosition):
#    chr13: start 16000000
#    chr14: start 16000000
#    chr15: start 17000000
#    chr21: start 5010000
#    chr22: start 1050000 
#    
#    chromosomeOrderList: ust gives the order in which the function processes the 
#    chromosomes. The list should only contain chromo names that are keys in chromosomeDict.
#    '''
#    
#    #Loop over chromosomes
#    # -- step1 get the ... 
#    # -- step2 get the pred-array
#    # -- step3 split in segments
#    
#    
#    for chromoName in chromosomeOrderList:
#        
#        #step1
#        fileNameModelPrediction = modelPredicitionFileNameDict[chromoName] #path to file containing the prediction for the model on the given chromosome
#        
#        genomeFileName = chromoName + ".txt"
#        fileGenome = rootGenome + genomeFileName #path to the genome/chromosome sequence
#
#        positionArray, predArray, refBaseList = readInProbsFromExternal(fileNameModelPrediction)
#        
#        #step2
#        getAllPositions_b = 1
#        exonicInfoBinaryFileName = ''
#        chromoNameBound = 100
#        startAtPosition, endAtPosition  = chromosomeDict[chromoName]
#
#        predArray, qualArray, samplePositions, sampleGenomeSequenceEncoded = getPredArrayFromExternal(getAllPositions_b = getAllPositions_b, samplePositions = 0, sampleGenomeSequence = '', positionsExternal= positionArray, probsExternal= predArray, refBaseListExternal = refBaseList,  fileGenome = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, chromoNameBound = chromoNameBound, startAtPosition =startAtPosition, endAtPosition = endAtPosition,   k= kModel, displacementOfExternal = displacementOfExternal)
#          
#getPredArrayFrqModel(getAllPositions_b, samplePositions,  sampleGenomeSequence, labelArray, fileGenome, exonicInfoBinaryFileName, chromoNameBound, startAtPosition, endAtPosition, resultsDictFrqModel, k, samplePositionsIndicatorArray = 0):
#         
#  
#        #step3            
#        genomeIdNameSeg = chromoName + '_seg' + str(int(segmentLength))
#        
##        startAtSegment = 0
##        if startAtSegmentDict:
##            
##            startAtSegment = startAtSegmentDict[chromoName]
#
#        #Create a directory if its not already there:
#        rootOutput_thisChromo = rootOutput +  chromoName + '/'
#        if not os.path.exists(rootOutput_thisChromo):
#
#            os.makedirs(rootOutput_thisChromo)
#            print("Directory " + rootOutput_thisChromo + " created. Output will be placed there.")
#        
#        splitExternalPredArrayInSegments(rootOutput = rootOutput_thisChromo,
#                                     modelFileName = modelFileName,
#                                     averageRevComplementary_b = averageRevComplementary_b,
#                                     genomeIdName = genomeIdNameSeg,
#                                     predArray = predArray, 
#                                     qualifiedArray = qualArray,
#                                     labelArray = sampleGenomeSequenceEncoded, 
#                                     sampledPositionsArray = samplePositions,
#                                     segmentLength = segmentLength)
#                                   
#  






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


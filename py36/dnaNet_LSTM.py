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
    
dnaNet_LSTM_v2: 26 mar '21; minor changes (now only 1 nrLSTMlayer param, nrLSTMLayer removed .. notice the capital L ...)  

dnaNet_LSTM_v3: id to dnaNet_LSTM_v2 except for these additions in allInOneWithDynSampling_ConvLSTMmodel:
1. Letting the call to encodeGenome (via fastReadGenome) output the chromoList, showing which chromo's it has read data for.

2. the logging in the runData file: 
A. of avoidChromo; in v2 the chr's to be avoided were pop'ed 
from the provided list (avoidChromo) so that when the final logging in runData was done, the list 
avoidChromo was empty. Here, in v3 we log both the initial list and its remains at the end when the 
training is done.
B. logging the chromoList as now output from encodeGenome (see 1)

3. Modifications to accomodate the split-exercise (in which training and validation/test are taken from separate files, eg two
disjoint sets of positions)

####################################################

Import module:

####################################################

import dnaNet_LSTM_v3 as dnaNet

#human
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 200
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]

#In anger (yeast):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 2000
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.7
nrEpochs = 10
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,15000000]
nrTestSamples = 100000
testDataInterval = [10000000,-12000000]



exonicInfoBinaryFileName  = ''
inclFrqModel_b = 0
insertFrqModel_b = 0
customFlankSize = 50
overlap = 0
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
#lengthWindows = [3,3]
#nrFilters = [64, 256] 
lengthWindows = [4]
nrFilters = [256] 
filterStride = 1
#lstm layers:
nrOfParallelLSTMstacks = 1 #parallel LSTMs
nrLSTMlayers = 1 #OBS: the run data file will record nrLSTMlayers as this number plus 1 if summarizingLSTMLayer_b == 1
summarizingLSTMLayer_b = 1
LSTMFiltersByLastConvFilters_b = 1
nrLSTMFilters = [-1]  #-1: just placeholder to be recorded in runData file
tryAveraging_b = 0
padding = 'valid'
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [50]

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
dropoutVal = 0.0
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.1 #default, but we use Adam here, so the value here isn't used
#learningRate = learningRate
chromoNameBound = 1000 #mouse:65 #yeast 1000
onlyOneRandomChromo_b = 0
avoidChromo = []  #['Homo sapiens chromosome 2', 'Homo sapiens chromosome 4', 'Homo sapiens chromosome 6', 'Homo sapiens chromosome 8', 'Homo sapiens chromosome 10', 'Homo sapiens chromosome 12', 'Homo sapiens chromosome 14', 'Homo sapiens chromosome 16', 'Homo sapiens chromosome 18', 'Homo sapiens chromosome 20', 'Homo sapiens chromosome 22'] 
on_binf_b = 1 
 
subStr = '1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00'
#subStr = '2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00'

learningRate = 0.001
modelName = 'modelLSTM_' + subStr
modelDescr = subStr

#With conv layer:

rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4P/trainTestSplit_80_20/"


#LSTM4P uses splitExercise_b = 1 and two files for loading genome data (one for the training and one for the validations):
fileGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/part1.txt"
fileGenome_forVal = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/part2.txt"
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(splitExercise_b = 1, genomeFileName_forVal = fileGenome_forVal, rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#yeast
rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4P/trainTestSplit_80_20/"

fileGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/split/part1/part1.txt"
fileGenome_forVal = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/split/part2/part2.txt"
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(splitExercise_b = 1, genomeFileName_forVal = fileGenome_forVal, rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#plain, not using the partitioned genome:    
rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/LSTM4R/trainTestSplit_70_30/"

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"
fileGenome = rootGenome +fileName
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




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
fileName = r"/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/GCF_000001405.38_GRCh38.p12_genomic_filter.fna" 
fileGenome = fileName

#Yeast genome

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"
fileGenome = rootGenome +fileName

#Single chromo:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"R64_chr1.txt"
fileGenome = rootGenome +fileName

#Droso:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/"
fileName = r"dmel-all-chromosome-r6.18.fasta"
fileGenome = rootGenome +fileName

#Zebrafish:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
fileName = r"GCF_000002035.6_GRCz11_genomic.fna"
fileGenome = rootGenome +fileName

#Mouse:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
fileName =  r"Mus_musculus.GRCm38.dna_sm.primary_assembly.fa"
fileGenome = rootGenome +fileName

Mouse, next-to-last ref assembly, mm9:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/"
fileName =  r"mm9.fa"
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
nrOfRepeats = 2
firstRepeatNr = 0
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 10
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 200000
trainDataInterval = [0,3000000]
nrTestSamples = 10000
testDataInterval = [3000000,3200000]

#In anger (human, mouse GRCm38):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 201
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]

In anger (mouse mm9):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 201
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,2800000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]

#In anger (yeast):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 100
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,15000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]


#In anger (droso):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 200
firstRepeatNr = 146 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,150000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]

#In anger (zebrafish):
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 200
firstRepeatNr = 0 #if = n > 0: loads in model from repeatNr n-1 
testDataIntervalIdTotrainDataInterval_b = 1
trainTestSplitRatio = 0.8
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 
trainDataInterval = [0,2000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]


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
lengthWindows = [3,3]
nrFilters = [64, 256] 
#lengthWindows = [4]
#nrFilters = [256] 
filterStride = 1
#lstm layers:
nrOfParallelLSTMstacks = 1 #parallel LSTMs
nrLSTMlayers = 1 #OBS: the run data file will record nrLSTMlayers as this number plus 1 if summarizingLSTMLayer_b == 1
summarizingLSTMLayer_b = 1
LSTMFiltersByLastConvFilters_b = 1
nrLSTMFilters = [-1]  #-1: just placeholder to be recorded in runData file
tryAveraging_b = 0
padding = 'valid'
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [50]

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
dropoutVal = 0.0
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.1 #default, but we use Adam here, so the value here isn't used
#learningRate = learningRate
chromoNameBound = 100 #mouse:65
onlyOneRandomChromo_b = 0
avoidChromo = ['Homo sapiens chromosome 2', 'Homo sapiens chromosome 4', 'Homo sapiens chromosome 6', 'Homo sapiens chromosome 8', 'Homo sapiens chromosome 10', 'Homo sapiens chromosome 12', 'Homo sapiens chromosome 14', 'Homo sapiens chromosome 16', 'Homo sapiens chromosome 18', 'Homo sapiens chromosome 20', 'Homo sapiens chromosome 22'] 
on_binf_b = 1 
 
#subStr = '1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00'
subStr = '2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00'

learningRate = 0.001
modelName = 'modelLSTM_' + subStr
modelDescr = subStr

#With conv layer:
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/zebrafish/on_GRCz11/trainTestSplit_80_20/"
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64/trainTestSplit_80_20/"
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18/trainTestSplit_80_20/"
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/trainTestSplit_80_20/"
#rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_mm9/trainTestSplit_80_20/"

rootOutput =  r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S2/trainTestSplit_80_20/"


labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#LSTM4S and LSTM4S2, uses avoidChromo
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(avoidChromo = avoidChromo, rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#restart:
firstIterNr = 0
firstRepeatNr = 97
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Only LSTM:
labelsCodetype = 1 #1: base pair prediction
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, labelsCodetype = labelsCodetype, convLayers_b = 0, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,   lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




#########################
#For test of sampling:
subStr = 'testSampling'

learningRate = 0.001
modelName = 'modelLSTM_' + subStr
modelDescr = subStr

rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/"

labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeIDmodel' 
 
testOfSamplingOnly_b = 1
firstRepeatNr = 0
samplingCountsTotal_train, samplingCountsTotal_test = dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, chromoNameBound = chromoNameBound, trainTestSplitRatio = trainTestSplitRatio, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, nrLSTMlayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b, testOfSamplingOnly_b = testOfSamplingOnly_b)


#Look at results
import numpy as np
import cPickle as pickle

loadfile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/testOfSampling_LSTM_samplingCountsTotal_train_repeatNr0.p'
t= pickle.load(open(loadfile,"rb"))
np.sum(t)

loadfile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/testOfSampling_LSTM_samplingCountsTotal_test_repeatNr0.p'
t= pickle.load(open(loadfile,"rb"))
np.sum(t)

loadfile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/testOfSampling_LSTM_trainSetIndicator.p'
t= pickle.load(open(loadfile,"rb"))
np.sum(t)

loadfile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM/testSampling/testOfSampling_LSTM_testSetIndicator.p'
t= pickle.load(open(loadfile,"rb"))
np.sum(t)




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!! FROM HERE ON THE COMMANDS ARE OLD AND WILL NEED SOME WORK.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


####################################################
# Only testing
####################################################

rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg19/"
#rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"


modelName = 'modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00'

testOnly_b = 1
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
learningRate = 0
firstRepeatNr = 62
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(testOnly_b = testOnly_b, rootOutput = rootOutput, usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



####################################################

#Merging a model merged with a k-mer model and training the combo:

####################################################


#For looking at the flow:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 0 
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
nrOfRepeats = 5
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
firstRepeatNr = 0 #loads in model from repeatNr ...!
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,3000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,-12000000]



#Modelling spec's. OBS: THE NAME REVEALS THAT FRQ MODEL IS INCLUDED
exonicInfoBinaryFileName  = ''
inclFrqModel_b = 1
insertFrqModel_b = 0
customFlankSize = 50
overlap = 0
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
#lengthWindows = [3,3]
#nrFilters = [64,256] 
lengthWindows = [4]
nrFilters = [256] 
filterStride = 1
#parallel LSTMs:
nrOfParallelLSTMstacks = 1
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [50]
#Nr of lstm layers:
nrLSTMlayers = 2 #26 March '21: not used (?), there isonly 1 LSTM layer in the one-sided makeConv1DLSTMmodel
tryAveraging_b = 1
padding = 'valid'


#set-up
dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
file = "frqModel_chr10_k4.txt"
#file = "frqModel_chr10_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
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



subStr = '_1Conv2LayerLstmAverageByConv_flanks50_win4_stride1_overlap0_dropout00_inclFrq'
learningRate = 0.001
modelName = 'ownSamples/human/inclRepeats/modelLSTM_' + subStr
modelDescr = subStr


#With conv layer:
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(usedThisModel = usedThisModel, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



####################################################

#Set up training schedule, model and run for the Onesided convo+LSTM:

####################################################


#For looking at the flow:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 0 
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
nrOfRepeats = 5
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
firstRepeatNr = 0 #116 #loads in model from repeatNr 115!
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
customFlankSize = 200
overlap = 0
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [3,3]
nrFilters = [64,256] 
#lengthWindows = [4]
#nrFilters = [256] 
filterStride = 1
#parallel LSTMs:
nrOfParallelLSTMstacks = 1
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [20]
#Nr of lstm layers:
nrLSTMlayers = 2 #26 March '21: not used (?), there isonly 1 LSTM layer in the one-sided makeConv1DLSTMmodel
tryAveraging_b = 0
padding = 'valid'


#set-up
dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
file = "frqModel_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
exclFrqModelFlanks_b = 0
##!!:
augmentWithRevComplementary_b = 1
##
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



subStr = '_oneSided_2Conv2LayerLstm_flanks200_win3_filters64And256_stride1_overlap0_dropout00'
learningRate = 0.001
modelName = '/modelLSTM_' + subStr
modelDescr = subStr


#With conv layer:
rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg38/"
outputEncodedOneHot_b = True
outputEncodedInt_b = False
oneSided_b = True
labelsCodetype = 0 #1: base pair type prediction
usedThisModel = 'makeOnesidedConv1DLSTMmodel'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(rootOutput = rootOutput, usedThisModel = usedThisModel, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, oneSided_b = oneSided_b, labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, tryAveraging_b= tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



'''



#THEANO_FLAGS='floatX=float32,device=cuda' 
#TENSORFLOW_FLAGS='floatX=float32,device=cuda' 

import os


#Outcommented when using colab
# #set this manually at very beginning of python session (dont set anything when using Hecaton; weill get you the GTX 1080, which has high enough compute capability)
# #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# #to prevent the process from taking up all the ram on the gpu upon start:
# #import tensorflow as tf
# import tensorflow as tf

# config = tf.ConfigProto(device_count = {'GPU': 0})
# #config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)

# #tf.device('/gpu:1')

# sess = tf.Session(config=config)
# #sess = tf.compat.v1.Session(config=config)
# tf.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
# #tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


#when using colab
import tensorflow as tf

#8/3 '22: all keras replaced by tensorflow.keras:
from tensorflow.keras import utils, backend 
#from keras import backend

from tensorflow.keras.models import Sequential, Model

#Conv1D
from tensorflow.keras.layers import Conv1D, Conv2D, Input, Dense, Dropout, AveragePooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling2D, MaxPooling2D, Flatten, Concatenate, Reshape #, merge
#Additional for LSTM
from tensorflow.keras.layers import LSTM, Activation, Bidirectional, concatenate, Lambda, multiply, Add, RepeatVector, Permute, Dot

#for colab
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#original:
#from tekeras.optimizers import SGD, RMSprop, Adam


from tensorflow.keras.models import model_from_json

from keras.utils.vis_utils import plot_model

from scipy.fftpack import fft, ifft

import numpy as np
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from random import shuffle

import pickle

from scipy.sparse import csr_matrix

#import graphviz
#import pydot

import frqModels as frqM 

import dnaNet_dataGen as dataGen #all smpling aso is here


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
    

###############################
## LSTM's
###############################

def makeLSTMmodel(sequenceLength,  nrLayers = 1, nrFilters = 4, outputSize = 4, goBackwardsOnRightSide_b = False, return_sequences=False, stateful=False, initialDropout_b = 0, dropout_b = 0, dropoutVal = 0.25, finalDenseLayers_b = 0, sizeHidden = [10]):
    '''
    network model
    sequenceLength = lenght of the sequence (number of letters)
    nrFilters (letterShape): shape of letter encoding, here default arrays of length 4
    outputSize = the size of the output layer, here 4 by default
    '''

    print('Build LSTM model...')

    inputs_left = Input(shape=(sequenceLength,nrFilters))
    inputs_right = Input(shape=(sequenceLength,nrFilters))

#    #all but the last lstm -layer returns a sequence (of the same length as the input)
#    for i in range(nrLayers -1):
#
#        inputs_left  = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_left)
#        inputs_right = LSTM(letterShape, return_sequences=True, stateful=stateful)(inputs_right)
#
#        print(inputs_left.shape)
#        print (inputs_right.shape)

    if initialDropout_b ==  1:
            
        inputs_left = Dropout(dropoutVal)(inputs_left)
        inputs_right = Dropout(dropoutVal)(inputs_right)
    
    if nrLayers > 1: 
    
        lstm_left_1  = LSTM(nrFilters, return_sequences=True, stateful=stateful)(inputs_left)
        lstm_right_1 = LSTM(nrFilters, return_sequences=True, stateful=stateful, go_backwards = goBackwardsOnRightSide_b)(inputs_right)
    
        print(lstm_left_1.shape)
        print (lstm_right_1.shape)
    
        if dropout_b ==  1:
            
                lstm_left_1 = Dropout(dropoutVal)(lstm_left_1)
                lstm_right_1 = Dropout(dropoutVal)(lstm_right_1)
            

    if nrLayers > 2: 
    
        lstm_left_2  = LSTM(nrFilters, return_sequences=True, stateful=stateful)(lstm_left_1)
        lstm_right_2 = LSTM(nrFilters, return_sequences=True, stateful=stateful)(lstm_right_1)
    
        print(lstm_left_2.shape)
        print (lstm_right_2.shape)
    
        if dropout_b ==  1:
            
                lstm_left_2 = Dropout(dropoutVal)(lstm_left_2)
                lstm_right_2 = Dropout(dropoutVal)(lstm_right_2)


    if nrLayers > 3: 

        lstm_left_3  = LSTM(nrFilters, return_sequences=True, stateful=stateful)(lstm_left_2)
        lstm_right_3 = LSTM(nrFilters, return_sequences=True, stateful=stateful)(lstm_right_2)
    
        print(lstm_left_3.shape)
        print (lstm_right_3.shape)
    
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
    
    if nrLayers > 1:
        goBackwardsOnRightSide_b = False
    #last lstm layer:        
    lstm_left  = LSTM(nrFilters, return_sequences=False, stateful=stateful)(left_firstPart)
    lstm_right = LSTM(nrFilters, return_sequences=False, stateful=stateful, go_backwards = goBackwardsOnRightSide_b)(right_firstPart)

    print("Left-hand shape after first LSTM part ", lstm_left.shape)
    print("Right-hand shape after first LSTM part ",lstm_right.shape)

    if dropout_b ==  1:
            
        lstm_left= Dropout(dropoutVal)(lstm_left)
        lstm_right = Dropout(dropoutVal)(lstm_right)

    
    #Concatenate the two LSTM-outputs:
    leftAndRight = concatenate([lstm_left, lstm_right], axis=-1, name = 'concat')
    
    print(leftAndRight.shape)

    if finalDenseLayers_b == 1:
        
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight.shape)

    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print(prediction.shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)


    print("... build model.")
        
    return model



def makeConv1DLSTMmodel(sequenceLength, letterShape, lengthWindows, nrFilters, nrLSTMLayers = 1, summarizingLSTMLayer_b = 1, LSTMFiltersByLastConvFilters_b = 1, nrLSTMFilters = [16], filterStride = 1, onlyConv_b = 0, nrOfParallelLSTMstacks = 1, finalDenseLayers_b = 0, sizeHidden = [10], paddingType = 'valid', outputSize = 4,  batchSize = 100, pool_b = 0, maxPooling_b = 0, poolAt = [2], poolSize = 2, poolStrides = 2, dropoutConvLayers_b = 1, dropoutLSTMLayers_b = 0, dropoutVal = 0.25, return_sequences=False, stateful=False, tryAveraging_b = 0):
    '''
    network model
    flankSize = lenght of the flanking sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    nrLSTMLayers: the number of LSTM layers returning sequences, but if desired (summarizingLSTMLayer_b = 1) a final summarizing LSTM layer is outputting the "central position"   
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
            convOutLeft = MaxPooling1D(pool_size = poolSize, strides = poolStrides)(convOutLeft)
            convOutRight = MaxPooling1D(pool_size = poolSize, strides = poolStrides)(convOutRight)
        else:
            convOutLeft = AveragePooling1D(pool_size = poolSize, strides = poolStrides)(convOutLeft)
            convOutRight = AveragePooling1D(pool_size = poolSize, strides = poolStrides)(convOutRight)

    if dropoutConvLayers_b ==  1:
        
        convOutLeft = Dropout(dropoutVal)(convOutLeft)
        convOutRight = Dropout(dropoutVal)(convOutRight)
        
    print("Left-hand shape after 1st convo ", convOutLeft.shape)
    print("Right-hand shape after 1st convo ",convOutRight.shape)

    
    for i in range(len(nrFilters)-1):    
    
        convOutLeft = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutLeft)
        convOutRight = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutRight)

        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                convOutLeft = MaxPooling1D(pool_size = poolSize, strides = poolStrides)(convOutLeft)
                convOutRight = MaxPooling1D(pool_size = poolSize, strides = poolStrides)(convOutRight)
            else:
                convOutLeft = AveragePooling1D(pool_size = poolSize, strides = poolStrides)(convOutLeft)
                convOutRight = AveragePooling1D(pool_size = poolSize, strides = poolStrides)(convOutRight)
                
                
        if dropoutConvLayers_b ==  1:
        
            convOutLeft = Dropout(dropoutVal)(convOutLeft)
            convOutRight = Dropout(dropoutVal)(convOutRight)
            
        print(convOutLeft.shape)
        print(convOutRight.shape)


    print("...by!")
    
    
    
    print("Left-hand shape after all convo's ",convOutLeft.shape)
    print("Right-hand shape after all convo's ", convOutRight.shape)
    
    
    if onlyConv_b == 1:
        
        flattenLeft = Reshape((-1,))(convOutLeft)
        print(flattenLeft.shape)
        flattenRight = Reshape((-1,))(convOutRight)
        print(flattenRight.shape)
        leftAndRight = concatenate([flattenLeft, flattenRight], axis = -1) 
        
    else:
        
        print("Then the lstm part ..." )
        
        if LSTMFiltersByLastConvFilters_b == 1:
            
            nrLSTMFilters = []
            for i in range(nrLSTMLayers):
                nrLSTMFilters.append(nrFilters[::-1][0])
                
        print("I'm using these filter sizes for the lstms ", nrLSTMFilters )
            
    
    #    lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
    #    lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
    #
    #    print(lstm_left_1.shape)
    #    print (lstm_right_1.shape)
    #
    #    #Concatenate the two LSTM-outputs:
    #    leftAndRight = concatenate([lstm_left_1, lstm_right_1], axis=-1)
    
        for j in range(nrOfParallelLSTMstacks):
            
            for i in range(nrLSTMLayers):
    
                if i == 0:
                    lstm_left_i  = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(convOutLeft)
                    lstm_right_i = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(convOutRight)
                else:
                    lstm_left_i  = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(lstm_left_i)
                    lstm_right_i = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(lstm_right_i)
                
    
                if dropoutLSTMLayers_b ==  1:
                    
                    lstm_left_i = Dropout(dropoutVal)(lstm_left_i)
                    lstm_right_i = Dropout(dropoutVal)(lstm_right_i)
    
                print("LSTM layer: ", i)
                print("Left-hand shape after after i'th LSTM", lstm_left_i.shape)
                print("Right-hand shape after i'th LSTM ",lstm_right_i.shape)
            
            
                       
        #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)
        #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
        #
        #    print(lstm_left_2.shape)
        #    print (lstm_right_2.shape)
        
            if summarizingLSTMLayer_b == 1:
                
                if tryAveraging_b == 1:
                    
                    #Obs i = nrLSTMLayers -1 at this point. So nrLSTMFilters[i] is the nr of filters in the last LSTM-layer
                    lstm_left_2  = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(lstm_left_i)  
                    lstm_right_2 = LSTM(nrLSTMFilters[i], return_sequences=True, stateful=stateful)(lstm_right_i)
                    
    #                lstm_left_2 = GlobalAveragePooling1D()(lstm_left_2)
    #                lstm_right_2 = GlobalAveragePooling1D()(lstm_right_2)
                    lstm_left_2 = Conv1D(kernel_size=10, strides=filterStride, filters=100, padding=paddingType, activation='relu')(lstm_left_2)
                    lstm_right_2 = Conv1D(kernel_size=10, strides=filterStride, filters=100, padding=paddingType, activation='relu')(lstm_right_2)
    
                    lstm_left_2 = Conv1D(kernel_size=10, strides=filterStride, filters=10, padding=paddingType, activation='relu')(lstm_left_2)
                    lstm_right_2 = Conv1D(kernel_size=10, strides=filterStride, filters=10, padding=paddingType, activation='relu')(lstm_right_2)
                    
                    lstm_left_2 = Reshape((-1,))(lstm_left_2)
                    lstm_right_2 = Reshape((-1,))(lstm_right_2)
    
                    
                else:
                    
                    #Obs i = nrLSTMLayers -1 at this point. So nrLSTMFilters[i] is the nr of filters in the last LSTM-layer                    
                    lstm_left_2  = LSTM(nrLSTMFilters[i], return_sequences=False, stateful=stateful)(lstm_left_i)  
                    lstm_right_2 = LSTM(nrLSTMFilters[i], return_sequences=False, stateful=stateful)(lstm_right_i)
                
            #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
            #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
                
                if dropoutLSTMLayers_b ==  1:
                    
                    lstm_left_2 = Dropout(dropoutVal)(lstm_left_2)
                    lstm_right_2 = Dropout(dropoutVal)(lstm_right_2)
    
            
                print("Left-hand shape after 2nd LSTM ",lstm_left_2.shape)
                print ("Right-hand shape after 2nd LSTM ", lstm_right_2.shape)
                
                #Concatenate the two LSTM-outputs, but flatten first:
                leftAndRight_j = concatenate([lstm_left_2, lstm_right_2], axis=-1)
            
            elif summarizingLSTMLayer_b == 0:
                #Concatenate the two LSTM-outputs, but first "flatten" the output (we're passing it to the final densely connected std NN layers below):
                lstm_left_i = Reshape((-1,))(lstm_left_i)
                lstm_right_i = Reshape((-1,))(lstm_right_i)
                leftAndRight_j = concatenate([lstm_left_i, lstm_right_i], axis=-1)
            
            if j == 0:
                
                leftAndRight = leftAndRight_j
                
            else: 
                
                leftAndRight = concatenate([leftAndRight, leftAndRight_j], axis = -1) 
                
        
            print("Shape of concatenated LSTM output ", leftAndRight.shape)
            
        
        print("Shape of LSTM-stacks output ", leftAndRight.shape)
    
    if finalDenseLayers_b == 1:
        
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight.shape)
    
    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print("Output shape ", prediction.shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)

    print("... Model's build.")
    
    return model



'''Here's a one-sided conv+LSTM, similar to the above bi-directional:'''
def makeOnesidedConv1DLSTMmodel(sequenceLength, letterShape, lengthWindows, nrFilters, filterStride = 1, onlyConv_b = 0, nrOfParallelLSTMstacks = 1, finalDenseLayers_b = 0, sizeHidden = [10], paddingType = 'valid', outputSize = 4,  batchSize = 100, pool_b = 0, maxPooling_b = 0, poolAt = [2], dropoutConvLayers_b = 0, dropoutVal = 0.25, return_sequences=False, stateful=False, tryAveraging_b = 0):
    '''
    
    network model

    One-sided conv+LSTM, similar to the bi-directional makeConv1DLSTMmodel.
    
    Obs: this one-sided version uses the left-hand part of the (left, right) split input 
    which is used in the bi-directional version. Ie: The model only 
    uses the "left" input.

    flankSize = lenght of the flanking sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the final dense layer (if finalDenseLayer_b = 1)
    outputSize = the size of the output layer, here 4
    '''

    print('Build Onesided Conv1d plus LSTM model...')    
    
    inputs_left = Input(shape=(sequenceLength,letterShape))
#    inputs_right = Input(shape=(sequenceLength,letterShape))

    print("First the 1d-convo, ba ...")

    convOutLeft = Conv1D(kernel_size=lengthWindows[0], strides=filterStride, filters=nrFilters[0], padding=paddingType, activation='relu')(inputs_left)

    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            convOutLeft = MaxPooling1D(convOutLeft)
        else:
            convOutLeft = AveragePooling1D(convOutLeft)


    if dropoutConvLayers_b ==  1:
        
        convOutLeft = Dropout(dropoutVal)(convOutLeft)
        
    print("Left-hand shape after 1st convo ", convOutLeft.shape)
    
    for i in range(len(nrFilters)-1):    
    
        convOutLeft = Conv1D(kernel_size=lengthWindows[i+1], strides=filterStride, filters=nrFilters[i+1], padding=paddingType, activation='relu')(convOutLeft)

        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                convOutLeft = MaxPooling1D(convOutLeft)
            else:
                convOutLeft = AveragePooling1D(convOutLeft)
                
        if dropoutConvLayers_b ==  1:
        
            convOutLeft = Dropout(dropoutVal)(convOutLeft)
            
        print(convOutLeft.shape)


    print("...by!")
    
    print("Left-hand shape after all convo's ",convOutLeft.shape)
    
    if onlyConv_b == 1:
        
        flattenLeft = Reshape((-1,))(convOutLeft)
        print(flattenLeft.shape)
        
        left = flattenLeft
        
    else:
        
        print("Then the lstm part ..." )
    
        prevLeft = convOutLeft

        for j in range(nrOfParallelLSTMstacks-1):
    
            lstm_left  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(prevLeft)
        
            if dropoutConvLayers_b == 1:
                lstm_left = Dropout(dropoutVal)(lstm_left)
        
            print("Left-hand shape after LSTM ", lstm_left.shape)

            prevLeft = lstm_left
            
            
    lstm_left  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(prevLeft)  
    if dropoutConvLayers_b ==  1:
        lstm_left = Dropout(dropoutVal)(lstm_left)

    left_j = lstm_left 
    print("Shape of  LSTM output ", left_j.shape)

    prevLayer = left_j
    if finalDenseLayers_b == 1:
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            left = Dense(sizeHidden[i], activation='relu', kernel_initializer='he_normal')(prevLayer)
            prevLayer = left
            
    print("Shape after final dense layer ", prevLayer.shape)
    
    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(prevLayer)

    print("Output shape ", prediction.shape)

    model = Model(inputs=inputs_left, outputs=prediction)

    print("... Model's build.")
    
    return model
    




def makeConv1DLSTMmodelFusedWithFrqModel(frqModelOutputSize,
                                         sequenceLength, 
                                         letterShape, 
                                         lengthWindows, 
                                         nrFilters, 
                                         filterStride = 1, 
                                         onlyConv_b = 0, 
                                         nrOfParallelLSTMstacks = 1, 
                                         finalDenseLayers_b = 0, 
                                         sizeHidden = [10], 
                                         paddingType = 'valid', 
                                         outputSize = 4,  
                                         batchSize = 100, 
                                         pool_b = 0, 
                                         maxPooling_b = 0, 
                                         poolAt = [2], 
                                         dropoutConvLayers_b = 1, 
                                         dropoutVal = 0.25, 
                                         return_sequences=False, 
                                         stateful=False, 
                                         tryAveraging_b = 0):
    '''
    network model
    flankSize = lenght of the flanking sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the final dense layer (if finalDenseLayer_b = 1)
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv1d plus LSTM model fused with frq-model ... ')    
    

#    inputs_left = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize,sequenceLength,letterShape))
#    inputs_right = Input(shape=(sequenceLength,letterShape), batch_shape = (batchSize, sequenceLength,letterShape))

    inputsFrqModel = Input(shape=(frqModelOutputSize,letterShape))

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
        
    print("Left-hand shape after 1st convo ", convOutLeft.shape)
    print("Right-hand shape after 1st convo ",convOutRight.shape)

    
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
            
        print(convOutLeft.shape)
        print(convOutRight.shape)


    print("...by!")
    
    
    
    print("Left-hand shape after all convo's ",convOutLeft.shape)
    print("Right-hand shape after all convo's ", convOutRight.shape)
    
    
    if onlyConv_b == 1:
        
        flattenLeft = Reshape((-1,))(convOutLeft)
        print(flattenLeft.shape)
        flattenRight = Reshape((-1,))(convOutRight)
        print(flattenRight.shape)
        leftAndRight = concatenate([flattenLeft, flattenRight], axis = -1) 
        
    else:
        
        print("Then the lstm part ..." )
    
    #    lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
    #    lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
    #
    #    print(lstm_left_1.shape)
    #    print (lstm_right_1.shape)
    #
    #    #Concatenate the two LSTM-outputs:
    #    leftAndRight = concatenate([lstm_left_1, lstm_right_1], axis=-1)
    
        for j in range(nrOfParallelLSTMstacks):
    
            lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutLeft)
            lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutRight)
        
            print("Left-hand shape after 1st LSTM ", lstm_left_1.shape)
            print("Right-hand shape after 1st LSTM ",lstm_right_1.shape)
            
            
        #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)
        #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
        #
        #    print(lstm_left_2.shape)
        #    print (lstm_right_2.shape)
        
            if tryAveraging_b == 1:
                
                lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)  
                lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
                
#                lstm_left_2 = GlobalAveragePooling1D()(lstm_left_2)
#                lstm_right_2 = GlobalAveragePooling1D()(lstm_right_2)
                lstm_left_2 = Conv1D(kernel_size=10, strides=filterStride, filters=100, padding=paddingType, activation='relu')(lstm_left_2)
                lstm_right_2 = Conv1D(kernel_size=10, strides=filterStride, filters=100, padding=paddingType, activation='relu')(lstm_right_2)

                lstm_left_2 = Conv1D(kernel_size=10, strides=filterStride, filters=10, padding=paddingType, activation='relu')(lstm_left_2)
                lstm_right_2 = Conv1D(kernel_size=10, strides=filterStride, filters=10, padding=paddingType, activation='relu')(lstm_right_2)
                
                lstm_left_2 = Reshape((-1,))(lstm_left_2)
                lstm_right_2 = Reshape((-1,))(lstm_right_2)

                
            else:
                
                lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_left_1)  
                lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_right_1)
            
        #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
        #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
            
        
            print("Left-hand shape after 2nd LSTM ",lstm_left_2.shape)
            print ("Right-hand shape after 2nd LSTM ", lstm_right_2.shape)
            
            #Concatenate the two LSTM-outputs:
            leftAndRight_j = concatenate([lstm_left_2, lstm_right_2], axis=-1)
            
            if j == 0:
                
                leftAndRight = leftAndRight_j
                
            else: 
                
                leftAndRight = concatenate([leftAndRight, leftAndRight_j], axis = -1) 
                
        
            print("Shape of concatenated LSTM output ", leftAndRight.shape)
            
        
        print("Shape of LSTM-stacks output ", leftAndRight.shape)
    
    if finalDenseLayers_b == 1:
        
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight.shape)
    
    
    print("Finally, merge in the output from the frq-model")
    #Merge in the output from the frq model:
    reshapedInputsFrqModel = Reshape((-1,))(inputsFrqModel)
    print(reshapedInputsFrqModel.shape)
    
    merge_outputs = Concatenate()([reshapedInputsFrqModel, leftAndRight])
    print(merge_outputs.shape)
        
    features = Reshape((-1,))(merge_outputs)
    print(features.shape)


    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(features)

    print("Output shape ", prediction.shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)

    print("... Model's build.")
    
    return model




#to define layer in makeConv1DLSTMmodelFusedWithEROmodel, which consists in 
#scaling the output tensors from parallel LSTM by the output of an ExonicRepeatOther-model
#and summing the components:
def scaleAndAdd(inputList):
    

    t1 = inputList[0]
    t2 = inputList[1]
#    print(t1.shape, t2.shape )
#    print t1[0], t2[0]
#    return backend.sum([t1[0][0]*t2[0][0], t1[0][1]*t2[0][1], t1[0][2]*t2[0][2]])

#    print t1, t2

#    return backend.sum(t1*t2, axis = -1, keepdims =True)
#    return backend.sum([t1[0]*t2[0], t1[1]*t2[1], t1[2]*t2[2]], axis = 1)
#    return tf.tensordot(t1,t2, axes = 0) 

#    output = backend.placeholder(shape = 512) #t2.shape) 
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
        
    print("Left-hand shape after 1st convo ", convOutLeft.shape)
    print("Right-hand shape after 1st convo ",convOutRight.shape)

    
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
            
        print(convOutLeft.shape)
        print(convOutRight.shape)


    print("...by!")
    
    
    
    print("Left-hand shape after all convo's ",convOutLeft.shape)
    print("Right-hand shape after all convo's ", convOutRight.shape)
        
    
    print("Then the lstm part ..." )
    
    lstmOutputList = []

#    lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
#    lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
#
#    print(lstm_left_1.shape)
#    print (lstm_right_1.shape)
#
#    #Concatenate the two LSTM-outputs:
#    leftAndRight = concatenate([lstm_left_1, lstm_right_1], axis=-1)

    for j in range(nrOfParallelLSTMstacks):

        lstm_left_1  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutLeft)
        lstm_right_1 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(convOutRight)
    
        print("Left-hand shape after 1st LSTM ", lstm_left_1.shape)
        print("Right-hand shape after 1st LSTM ",lstm_right_1.shape)
        
    #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_left_1)
    #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=True, stateful=stateful)(lstm_right_1)
    #
    #    print(lstm_left_2.shape)
    #    print (lstm_right_2.shape)
    
    
        lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_left_1)
        lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(lstm_right_1)
        
    #    lstm_left_2  = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutLeft)
    #    lstm_right_2 = LSTM(nrFilters[::-1][0], return_sequences=False, stateful=stateful)(convOutRight)
        
    
        print("Left-hand shape after 2nd LSTM ",lstm_left_2.shape)
        print ("Right-hand shape after 2nd LSTM ", lstm_right_2.shape)
        
        #Concatenate the two LSTM-outputs:
        leftAndRight_j = concatenate([lstm_left_2, lstm_right_2], axis=-1)

        if j == 0:

            lstmOutputList = leftAndRight_j
            
        else:             
           
            lstmOutputList = concatenate([lstmOutputList, leftAndRight_j], axis = 1)
            
        print("iter, shape:", j,lstmOutputList.shape)
#        lstmOutputList.append(leftAndRight_j)        

#    lstmOutputList = Input(lstmOutputList)
#
#    lstmOutputList = Reshape(( leftAndRight_j.shape[0], nrOfParallelLSTMstacks, leftAndRight_j.shape[1] ), input_shape =  lstmOutputList.shape)(lstmOutputList)

    newSize = int(lstmOutputList.shape[1]/nrOfParallelLSTMstacks) 
    print("newSize ", newSize)

    lstmOutputList = Reshape((  nrOfParallelLSTMstacks, newSize ), input_shape =  lstmOutputList.shape)(lstmOutputList) #convOutLeft.shape[1] repl'ed 512

    print (lstmOutputList.shape)


    print("Next get the ERO output ..")
    
#    eroOutput = Model((inputs_left, inputs_right), eroModel.output)
    eroOutput = eroModel([inputs_left, inputs_right])
    print (eroOutput.shape, eroOutput.dtype)
    
    print(eroOutput, [eroOutput], "ERO ouput")
        
    print(".. and scale the LSTMs with the ERO output:")
#    scaleAndAddLayer = Lambda(function = scaleAndAdd, output_shape =  leftAndRight_j.shape)
#    mergedInput = tf.concat([eroOutput, lstmOutputList], axis=0)
#    print (mergedInput.shape, mergedInput.dtype)
#    leftAndRight = scaleAndAddLayer([eroOutput[0], lstmOutputList[0]])
    
    eroOutputRepeated = RepeatVector(newSize)(eroOutput) #newSize repl'ed 512
    print (eroOutputRepeated.shape)
#    x = eroOutputRepeated.eval(session = sess)
#    tf.Print(x)
    #this should perform a transpose:
    eroOutputRepeatedT = Permute((2, 1), input_shape = eroOutputRepeated.shape)(eroOutputRepeated)
    print (eroOutputRepeatedT.shape)
#    y = eroOutputRepeatedT.eval(session = sess)
#    tf.Print(y)
    
    tf.Print(eroOutputRepeatedT, [eroOutputRepeatedT], "ERO ouput, repeated and transposed")
    
#    mergedInput = Input([eroOutputRepeatedT, lstmOutputList])
    leftAndRight = multiply([eroOutputRepeatedT, lstmOutputList])
        
#    leftAndRight = Add()(leftAndRight)
                
    print("Shape after multiply-layer ", leftAndRight.shape)

    leftAndRight = Lambda(function = summation, output_shape = (512,))(leftAndRight)
    
#    leftAndRight = Reshape((512 ), input_shape =  leftAndRight.shape)(leftAndRight)
    
    print("Shape of resulting output ", leftAndRight.shape)
    
    if finalDenseLayers_b == 1:
        
        nrDenseLayers = len(sizeHidden)
        for i in range(nrDenseLayers):
            
            leftAndRight = Dense(sizeHidden[i], activation='relu')(leftAndRight)
            
    print("Shape after final dense layer ", leftAndRight.shape)
    
    # And add a softmax on top
    prediction = Dense(outputSize, activation='softmax')(leftAndRight)

    print("Output shape ", prediction.shape)

    model = Model(inputs=[inputs_left, inputs_right], outputs=prediction)

    print("... Model's build.")
    
    return model





def allInOneWithDynSampling_ConvLSTMmodel(rootOutput =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/on_hg19/",
                                          nrOuterLoops = 1,
                                          firstIterNr = 0,
                                          nrOfRepeats = 1,
                                          firstRepeatNr = 0,
                                          loss = "categorical_crossentropy", 
                                          usedThisModel = 'makeConv1DLSTMmodel', #set this manually if restarting
                                          onHecaton_b = 0,
                                          convLayers_b = 1,
                                          oneSided_b = False,
                                          onlyConv_b = 0,
                                          leftRight_b = 1,
                                          fusedWitEROmodel_b = 0,
                                          eroModelFileName = '',
                                          nrOfParallelLSTMstacks = 1,
                                          finalDenseLayers_b = 0, #if 1 set the hiddenUnits param
                       learningRate = 0.01,
                       momentum = 0.0,
                       trainTestSplitRatio = 0.8,
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
            chromoNameBound = 1000,
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
            nrLSTMlayers = 1, 
            summarizingLSTMLayer_b = 1, 
            LSTMFiltersByLastConvFilters_b = 1, 
            nrLSTMFilters = [16],
            tryAveraging_b = 0,
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
            testOfSamplingOnly_b = 0, 
            splitExercise_b = 0,
            genomeFileName_forVal = '' #only used in split-exercise, ie when splitExercise_b = 1 
            ):
    
    '''
        labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding and not including the frq model 
                (inclFrqModel_b = 0).
                
        trainTestSplitRatio: allows to split test and train data in two disjoint sets; OBS: current only works for testDataIntervalIdTotrainDataInterval_b > 0.  

        TO DO: 
        * extend the one-sided version to also work for the testDataIntervalIdTotrainDataInterval_b = 0 case. 
        * make the trainTestSplitRatio wotk also in case testDataIntervalIdTotrainDataInterval_b = 0
    '''
    
    #out-commented 9 March '22, not in use pt            
#    if on_binf_b == 1:
#        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
#        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
#    else:
#        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
#        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"

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
                avoidChromoAsInput = [chromo for chromo in avoidChromo] #fastReadGenome pop's from the list avoidChromo, so it should end empty! 
                if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
                    genomeArray, repeatArray, exonicArray, genomeString, chromoList = dataGen.encodeGenome(fileName = genomeFileName, chromoNameBound = chromoNameBound, avoidChromo= avoidChromo, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0, inclChromoListInOutput_b = 1)
                    lGenome = len(genomeArray)
                    genomeSeqSourceTrain = 'Potentially read data from whole genome (chromo\'s concatenated, if any), see chromoList in the runData log file. '
                    print('Read in data for these (chromos, len(chromo)): ', chromoList)
                    #In case we do the split exercie, in which the modle is train on one part of the genome and predicted on the remainder, 
                    #we use the latter also for the validations carrie dout at the end of each repeat ('round') of the training (Addition, 9 March '22):                     
                    if splitExercise_b == 1:
                        print("I'm doing the split-exercise, and will be using this part of the splitted genome for validations: ", genomeFileName_forVal)
                        genomeArray_forVal, repeatArray_forVal, exonicArray_forVal, genomeString_forVal, chromoList_forVal = dataGen.encodeGenome(fileName = genomeFileName_forVal, chromoNameBound = chromoNameBound, avoidChromo= avoidChromo, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0, inclChromoListInOutput_b = 1)
                        lGenome_forVal = len(genomeArray_forVal)
                        print('Read in validation-data for these (chromos, len(chromo)): ', chromoList_forVal)
                    
                elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                    genomeDictArray, repeatInfoDictArray, exonicInfoDictArray, genomeDictString = dataGen.encodeGenome(fileName = genomeFileName, avoidChromo = avoidChromo, chromoNameBound = chromoNameBound, exonicInfoBinaryFileName  = exonicInfoBinaryFileName ,  startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1)
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
                        
                if insertFrqModel_b != 1 and leftRight_b == 0:
                
                    #We need to split the data in the part input to the conv/lstm layer and 
                    #the part which is the output of frq model; the same is done for the
                    #test data (below):
                    sizeInputConv = 2*(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel)
                    
                    Xconv = np.zeros(shape = (batchSize, sizeInputConv, letterShape))
                    Xfrq = np.zeros(shape = (batchSize, 1, letterShape))
                    
                elif insertFrqModel_b != 1 and leftRight_b == 1:
                
                    #We need to split the data in the part input to the conv/lstm layer and 
                    #the part which is the output of frq model; the part fro the conv/lstm consists
                    #in a left and a right hand side; the same is done for the
                    #test data (below):
                    
                    if augmentWithRevComplementary_b == 0:
                        Xleft = np.zeros(shape = (batchSize, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                        Xright = np.zeros(shape = (batchSize, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                    else:
                        Xleft = np.zeros(shape = (2*batchSize, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                        Xright = np.zeros(shape = (2*batchSize, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                    
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
                
                frqModelDict = frqM.getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)
                                         
                                
                
    
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
            #in parallel with the fit_generator call below (and stops when that is done)
            #For the purpose of separating train and test data we pass in a boolean array indicating
            #the indices to sample; we generate this array based on the set trainTestSplitRatio (to 
            #save memory we use boolean arrays rather than just the set of indices; this implies an extra if-clause
            #in the sampling, but with a typical 80-20 split, this should be of small concern).
            #Modified 9 March '22 to accomodate the split-exercise (splitExercise_b =1)
            if splitExercise_b != 1:
                
                trainSetIndicator = np.zeros(shape = lGenome, dtype = 'int8')
                testSetIndicator = np.zeros(shape = lGenome, dtype = 'int8')            
                for i in range(lGenome):
                    
                    #0-1 toss with prob = trainTestSplitRatio:
                    ind = np.random.binomial(1, p= trainTestSplitRatio)
                    trainSetIndicator[i] = ind
                    testSetIndicator[i] = 1 - ind
            elif splitExercise_b == 1:
                
                trainSetIndicator = np.ones(shape = lGenome, dtype = 'int8')
                testSetIndicator = np.ones(shape = lGenome_forVal, dtype = 'int8')     
                trainTestSplitRatio = float(lGenome_forVal)/lGenome
            
            print("Train-test split sizes (nr of positions of genome array): ", np.sum(trainSetIndicator), np.sum(testSetIndicator) )
                
            #Modified 9 March '22 to accomodate the split-exercise (splitExercise_b =1):    
            def myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b):
               
                if testOfSamplingOnly_b == 0: #standard use case
                
                    while 1:
                        
                        if forTrain_b == 1: #for train set
                            X,Y = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = trainSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                        else: #for validation/test set                                         
                        
                            if splitExercise_b != 1:
                                X,Y = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = testSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                            else:
                                X,Y = dataGen.genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray_forVal, repeatArray = repeatArray_forVal, exonicArray = exonicArray_forVal, indicatorArray = testSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                                             genomeString = genomeString_forVal, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
                                
    
        #                sizeInput = X.shape[1]
                        if inclFrqModel_b == 1  and insertFrqModel_b != 1 and leftRight_b == 0:
                
                            Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                            Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                            Xfrq[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
                            
                            yield([Xfrq, Xconv],Y)
                            
                        elif inclFrqModel_b == 1  and insertFrqModel_b != 1 and leftRight_b == 1:
                
                            Xleft = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :].copy()
                            Xright = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :].copy()
                            #and reverse it:
                            Xright = np.flip(Xright, axis = 1)
                            
                            Xfrq[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :].copy()
                            
                            
                            yield([Xfrq, Xleft, Xright],Y)
                            
               
                        
                        elif inclFrqModel_b == 0 and leftRight_b == 0:
                            
                            yield(X,Y)
                        
                        elif oneSided_b == 0 and inclFrqModel_b == 0 and leftRight_b == 1:
                            
    #                        Xleft[:, :(customFlankSize+overlap), :] = X[:, :(customFlankSize + overlap) , :]
                            Xleft = X[:, :(customFlankSize + overlap) , :].copy()
                            Xright = X[:, (customFlankSize - overlap):, :].copy()
                            #and reverse it:
                            Xright = np.flip(Xright, axis = 1)
        
                            yield([Xleft, Xright], Y)
                            
    #                        print "X left shape", Xleft.shape
    #                        print "X right shape", Xright.shape
                            
                        elif oneSided_b == 1 and inclFrqModel_b == 0 and leftRight_b == 1:
                            
    #                        Xleft[:, :(customFlankSize+overlap), :] = X[:, :(customFlankSize + overlap) , :]
                            Xleft = X[:, :(customFlankSize + overlap) , :].copy()
        
                            yield(Xleft, Y)
                            
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


        #This first part is only for testing the sampling; after this section the code
        #is for the training and testing of the model and that only      
        if testOfSamplingOnly_b == 1: #for testing the sampling
        
            #Record the settings for the sampling test:
            testFileName = rootOutput + 'bigLoopIter' + str(n)
            
            #Write run-data to txt-file for documentation of the run:
            runDataFileName = testFileName + '_runData_samplingTestGeneratorLSTM.txt'
            runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
            
            s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
            s += "Test of sampling generator for LSTM\n" 
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
                #Following line moved her from 'if onlyOneRandomChromo_b == 1:' clause on 26 march 2021. Purpose: the training restricted to the 'odd numbered chromos'
                #7/3 '22: changed to avoidChromoAsInput, since avoidChromo will be pop'ed down to end up empty by fastReadGenome    
                s += "Data from these chromosomes to be avoided: " +  str(avoidChromoAsInput)  + "\n" 
                if onlyOneRandomChromo_b == 1:
                    s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                    s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
#                    s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
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
            s += "data were sampled from this chromoList (shows also the length of each chromo as read in): " + str(chromoList) + "\n" #changed from avoidChromo on 7/3 '22, since avoidChromo will be pop'ed down to end up empty by fastReadGenome
            if splitExercise_b == 1:
                s += "split-exercise: validation data were sampled from this chromoList (shows also the length of each chromo as read in): " + str(chromoList_forVal) + "\n" #added on 9/3 '22
            s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
            s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
            s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n"
            s += "orig avoidChromo list should be empty: " + str(avoidChromo)  + "\n" #TExt changed from avoidChromo on 7/3 '22;  avoidChromo will be pop'ed down to end up empty by fastReadGenome 
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
            history = idNet.fit_generator(myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
            #Now do the sampling; we call the sampling through each repeat (as in training) and test the sampling after each repeat (as in testing)  
            for k in range(firstRepeatNr, nrOfRepeats):
                
                print("Sampling testing, now at repeat ", k )
                
#                #read in history if start at repeat nr > 0:
#                if k == firstRepeatNr and k > 0:
#                    loadfile = rootOutput + r'/testOfSampling_LSTM_samplingCountsTotal_train_repeatNr' + str(k-1) + '.p'
#                    samplingCountsTotal_train = pickle.load(open(loadfile,"rb"))
#                    loadfile = rootOutput + r'/testOfSampling_LSTM_samplingCountsTotal_test_repeatNr' + str(k-1) + '.p'
#                    samplingCountsTotal_test = pickle.load(open(loadfile,"rb"))
                    
                
                #First part: sampling as in training
                samplingCountsThisRepeat_train = 0*samplingCountsThisRepeat_train #reset
                
                
                for n in range(nrEpochs):
                    print("epoch ", n)
#                    for s in range(stepsPerEpoch):
#                        print "step ", s
                        
#                        i = 0
#                        for (x,y,z) in myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b):
#
#                            i += 1
#                            if i == batchSize:
#                                break 
#                    pred = idNet.predict(myGenerator(customFlankSize,batchSize, inclFrqModel_b, labelsCodetype, forTrain_b), batch_size = batchSize)
                    gen = myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b = -1)
                    pred = idNet.predict_generator(gen, steps= stepsPerEpoch)
                    
                    pred = pred.flatten()
                    pred = pred.astype(np.int64, casting='unsafe', copy=False) #MUST be int64!
                    
#                    print pred
                    
                    for u in pred:

                        samplingCountsThisRepeat_train[u] += 1
                
                samplingCountsTotal_train += samplingCountsThisRepeat_train
                
                
                #2nd part: Now get the sampling for the test part:
                samplingCountsThisRepeat_test = 0*samplingCountsThisRepeat_test #reset
                
                gen = myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b = 0)
                pred = idNet.predict_generator(gen, steps = np.int(float(nrTestSamples)/batchSize))
                    
                pred = pred.flatten()
                pred = pred.astype(np.int64, casting='unsafe', copy=False) #MUST be int64!
                
#                    print pred
                
                for u in pred:

                    samplingCountsThisRepeat_test[u] += 1
                
                samplingCountsTotal_test += samplingCountsThisRepeat_test                
                
                
                #dump result for selected repeats; we take though only those accumulated since last dump; and dump the results as sparse:
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
                s += "train and test from separate parts/files (split-exercise), splitExercise_b: " + str(splitExercise_b) + "\n"
                s += "Train data taken from (only in split-exercise ie if splitExercise_b = 1): "  + genomeFileName_forVal + "\n"
                s += "exonicInfoBinaryFileName: " + exonicInfoBinaryFileName + "\n"
                s += 'inclFrqModel_b: ' + str(inclFrqModel_b) + "\n"
                s += 'frqModelFileName: ' + frqModelFileName + "\n"  
                s += "Letters are one-hot encoded" + "\n"
                s += "Labels are encoded as type" + str(labelsCodetype) + "\n"
                #Following line moved her from 'if onlyOneRandomChromo_b == 1:' clause on 26 march 2021. Purpose: the training restricted to the 'odd numbered chromos'
                #7/3 '22: changed to avoidChromoAsInput, since avoidChromo will be pop'ed down to end up empty by fastReadGenome    
                s += "Data from these chromosomes to be avoided: " +  str(avoidChromoAsInput)  + "\n" 
                if onlyOneRandomChromo_b == 1:
                    s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
                    s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
#                    s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
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
            s += "data were sampled from this chromoList (shows also the length of each chromo as read in): " + str(chromoList) + "\n" #changed from avoidChromo on 7/3 '22, since avoidChromo will be pop'ed down to end up empty by fastReadGenome
            s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
            s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
            s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n"
            s += "orig avoidChromo list should be empty: " + str(avoidChromo)  + "\n" #Text changed from avoidChromo on 7/3 '22;  avoidChromo will be pop'ed down to end up empty by fastReadGenome
            s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
            s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
            s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
            s += 'learningRate: ' + str(learningRate) + "\n"
            s += 'batchSize: ' + str(batchSize) + "\n"
            s += 'dropout_b: ' + str(dropout_b) + "\n"
            s += 'dropoutVal: ' + str(dropoutVal) + "\n"
            s += 'tryAveraging_b: ' + str(tryAveraging_b) + "\n"
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
            s = 'oneSided_b: ' + str(oneSided_b) + "\n" 
            s += 'convLayers_b: ' + str(convLayers_b) + "\n"
            s += 'lengthWindows: ' + str(lengthWindows)  + "\n" 
            s += 'hiddenUnits: ' + str(hiddenUnits)  + "\n" 
            s += 'nrFilters: ' + str(nrFilters)  + "\n" 
            s += 'filterStride: ' + str(filterStride)  + "\n" 
            s += 'nrLSTMlayers: ' + str(nrLSTMlayers + summarizingLSTMLayer_b)  + "\n"       
            s += 'summarizingLSTMLayer_b: ' + str(summarizingLSTMLayer_b)  + "\n" 
            s += 'LSTMFiltersByLastConvFilters_b: ' + str(LSTMFiltersByLastConvFilters_b)  + "\n"       
            s += 'nrLSTMFilters: ' + str(nrLSTMFilters)  + "\n" 
            s += 'nrOfParallelLSTMstacks: ' + str(nrOfParallelLSTMstacks)
            
            runDataFile.write(s)
            
            s = '' #reset
            runDataFile.write(s + "\n") #insert blank line
        
            runDataFile.close()
            #Write run-data to txt-file for documentation of the run: DONE


        #Run series of repeated training-and-testing sessions each consisting in nrEpochs rounds:
        for k in range(firstRepeatNr, nrOfRepeats):       
                    
    
            #in first outer-iteration build the model; thereafter reload the latest stored version (saved below)
            if n == 0 and k == 0 and testOnly_b == 0: 
        
                if convLayers_b == 0 and fusedWitEROmodel_b == 0 and inclFrqModel_b == 0:
            
                    net = makeLSTMmodel(sequenceLength = customFlankSize + overlap, nrLayers = nrLSTMlayers, letterShape = letterShape,  outputSize = sizeOutput, batchSize = batchSizeReal, dropout_b = dropout_b, dropoutVal = dropoutVal )
                    
                    usedThisModel = 'makeLSTMmodel'
                    
                elif convLayers_b > 0 and fusedWitEROmodel_b == 0 and inclFrqModel_b == 0:
                    
                    if not oneSided_b:
                        if onlyConv_b != 1 or (onlyConv_b == 1 and leftRight_b == 1):
                        
                            net = makeConv1DLSTMmodel(sequenceLength = customFlankSize + overlap, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, filterStride = filterStride, onlyConv_b = onlyConv_b, nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, nrLSTMLayers = nrLSTMlayers, summarizingLSTMLayer_b = summarizingLSTMLayer_b, LSTMFiltersByLastConvFilters_b = LSTMFiltersByLastConvFilters_b, nrLSTMFilters = nrLSTMFilters, finalDenseLayers_b = finalDenseLayers_b, sizeHidden = hiddenUnits, outputSize = sizeOutput,  batchSize = batchSizeReal, tryAveraging_b = tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutConvLayers_b = dropout_b, dropoutVal = dropoutVal )
                    
                            usedThisModel = 'makeConv1DLSTMmodel'
                        
                        elif onlyConv_b == 1 and leftRight_b != 1:
                        
                            net = makeConv1Dmodel(sequenceLength = sizeInput, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, hiddenUnits = hiddenUnits, outputSize = sizeOutput, padding = padding, pool_b = pool_b, poolStrides = poolStrides, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b)
            
                            usedThisModel = 'makeConv1Dmodel'
                            
                    else: #if oneSided_b
                        net = makeOnesidedConv1DLSTMmodel(sequenceLength = customFlankSize + overlap, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, filterStride = filterStride, onlyConv_b = onlyConv_b, nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, finalDenseLayers_b = finalDenseLayers_b, sizeHidden = hiddenUnits, outputSize = sizeOutput,  batchSize = batchSize, tryAveraging_b = tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutConvLayers_b = dropout_b, dropoutVal = dropoutVal )
                        usedThisModel = 'makeOnesidedConv1DLSTMmodel'
                                    
                        
                elif convLayers_b > 0 and fusedWitEROmodel_b == 0 and inclFrqModel_b == 1:
                    
                    if onlyConv_b != 1 or (onlyConv_b == 1 and leftRight_b == 1):
                    
                        net = makeConv1DLSTMmodelFusedWithFrqModel(frqModelOutputSize = 1, sequenceLength = customFlankSize + overlap, letterShape = letterShape, lengthWindows = lengthWindows, nrFilters= nrFilters, filterStride = filterStride, onlyConv_b = onlyConv_b, nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, finalDenseLayers_b = finalDenseLayers_b, sizeHidden = hiddenUnits, outputSize = sizeOutput,  batchSize = batchSizeReal, tryAveraging_b = tryAveraging_b, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, dropoutConvLayers_b = dropout_b, dropoutVal = dropoutVal )
                
                        usedThisModel = 'makeConv1DLSTMmodelFusedWithFrqModel'
                    
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
        

            if testOnly_b == 0:#just means that we're running a regular training/testing session
                
                forTrain_b = 1
                history = net.fit_generator(myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b), steps_per_epoch= stepsPerEpoch, epochs=nrEpochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=2, workers=1, use_multiprocessing=False,  initial_epoch=1)
           
    
                if save_model_b == 1:
                     
                    json_string = net.to_json()
                    open(modelFileName + '_repeatNr' + str(k), 'w').write(json_string)
                    net.save_weights(modelFileName + '_repeatNr' + str(k) + '.h5',overwrite=True)
    
            
                # list all data in history
                print(history.history.keys())
                
                #dump the info:
                dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_acc_vs_epoch.p'
                pickle.dump( history.history['accuracy'], open(dumpFile, "wb") )
                dumpFile = modelFileName + '_repeatNr' + str(k) + '_training_loss_vs_epoch.p'
                pickle.dump( history.history['loss'], open(dumpFile, "wb") )
                
                
                # summarize history for accuracy
                plt.figure()
                plt.plot(history.history['accuracy'])
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
                historyTotal['acc'].extend(history.history['accuracy'])
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
            print("Now testing ...")
            if testDataIntervalIdTotrainDataInterval_b == 0:
               
                if inclFrqModel_b == 1:
                        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                            flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
        
                    if insertFrqModel_b != 1 and leftRight_b == 0:
                        
                        #Split the test data as the training data:
                        nrOfTestSamples = Xt.shape[0]
                        Xconv_t = np.zeros(shape = (nrOfTestSamples, sizeInputConv, letterShape))
                        Xfrq_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
            
                        Xconv_t[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
                        Xconv_t[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
                        Xfrq_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
                        
                    
                    elif insertFrqModel_b != 1 and leftRight_b == 1:
                        
                        #Split the test data as the training data:
                        nrOfTestSamples = Xt.shape[0]
                        Xt_left = np.zeros(shape = (nrOfTestSamples, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                        Xt_right = np.zeros(shape = (nrOfTestSamples, customFlankSize + overlap - exclFrqModelFlanks_b*flankSizeFrqModel, letterShape))
                        Xfrq_t = np.zeros(shape = (nrOfTestSamples, 1, letterShape))
                        
                        
                        Xt_left = Xt[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :].copy()
                        Xt_right = Xt[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :].copy()
                        #and reverse it:
                        Xt_right = np.flip(Xt_right, axis = 1)
                        Xfrq_t[:, 0, :] = Xt[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :].copy()
                        
                        
         
                    
                else:
        
                    #Read in the test data we avoid the chromos used for training:    
                    avoidChromo.append(genomeSeqSourceTrain) ##to avoid getting test data from the same chromo as the training and validation data 
                    Xt,Yt, genomeSeqSourceTest = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
                                                              flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelDict = frqModelDict, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b,  
                                                              onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1], shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentTestDataWithRevComplementary_b, getFrq_b = 0)
        
                    if leftRight_b == 1:
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
                    
                    elif insertFrqModel_b == 0 and leftRight_b == 0:
                        
                        score, acc = net.evaluate([Xfrq_t, Xconv_t], Yt, batch_size=batchSizeReal, verbose=1)
                    
                    elif insertFrqModel_b == 0 and leftRight_b == 1:
                        
                        score, acc = net.evaluate([Xfrq_t, Xt_left, Xt_right], Yt, batch_size=batchSizeReal, verbose=1)
                    
                else:
                    
                    score, acc = net.evaluate([Xt_left, Xt_right],Yt, batch_size=batchSizeReal, verbose=1)
                    
                    
            elif testDataIntervalIdTotrainDataInterval_b == 1: #we test using the dynamic sampling
                
                if splitExercise_b != 1:
                    print("In test: Test data interval id to train data interval!")
                    print("But: train-test split ratio set to: %f" % trainTestSplitRatio)
                else:
                    print("Split-exercise. The data for this test/validation are taken from another source than the train data")
                    print("Train-test split ratio set to: %f" % trainTestSplitRatio)
                    
                forTrain_b = 0
                score, acc = net.evaluate_generator(myGenerator(customFlankSize,batchSize, oneSided_b, inclFrqModel_b, insertFrqModel_b, labelsCodetype, forTrain_b), steps = np.int(float(nrTestSamples)/batchSize))
                    

                
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
            
   



#All in one run
#OBS: needs updating; not structured like allInOneWithDynSampling_ConvLSTMmodel
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
    lengthWindows = 3, 3, 4
    nrFilters = 20, 30, 40
    nrHiddenUnits = 10

  
    if genSamples_b > 0.5: #generate a set of random data acc to the sizes set

        #if a genomeFileName is specified, use that genome:
        if len(genomeFileName) > 0:
            fromGenome_b = 1
        else:
            fromGenome_b = 0
    
        X,Y, genomeSeqSourceTrain = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, genRandomSamples_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTrainSamples, startAtPosition = trainDataInterval[0], endAtPosition = trainDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        sizeInput = X.shape[1]
        print("Shape train set: ", X.shape)
        print("Shape train set labels: ", Y.shape)
        print("sizeInput: %d " % sizeInput)

        avoidChromo.append(genomeSeqSourceTrain) #to avoid getting val data from the same chromo as the training data 
        Xv,Yv, genomeSeqSourceVal = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrValSamples, startAtPosition = valDataInterval[0], endAtPosition = valDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        
        avoidChromo.append(genomeSeqSourceVal) ##to avoid getting test data from the same chromo as the training and validation data 
        Xt,Yt, genomeSeqSourceTest = dataGen.genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = nrTestSamples, startAtPosition = testDataInterval[0], endAtPosition = testDataInterval[1],  flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
        

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
        
            
            X,Y = dataGen.getData2(fname, letterShape, sizeOutput, loadRecsInterval = trainDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
            sizeInput = X.shape[1]
                    
            Xv,Yv = dataGen.getData2(vname, letterShape, sizeOutput, loadRecsInterval = valDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)      
                
            Xt,Yt = dataGen.getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
        
    
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
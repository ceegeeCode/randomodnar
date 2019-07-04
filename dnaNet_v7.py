# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk
"""


'''


Version notes:
* 14 Aug 18: 
- dyn sampling can be done "transformation style"; only impl'ed in conv-dyn fct
- test data are read in after the training is carried out; only impl'ed in conv-dyn fct
- model is saved before plotting and testing; only impl'ed in conv-dyn fct

- BUT: the transformation style dyn sampling seems 1) to corrupt the benefit of the frq model
  and 2) not to save any RAM/memory when running 

############################################################
    
    
Usage:
    
    OBS: this is a big "mess", both this usage section and the code ... good and not-so-good in a 
    big pile :--) I'll try and split it out in code dedicated for the various models (MLP, 
    convo etc) asap. 

############################################################
    
import dnaNet_v7 as dnaNet

root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"

rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/from_AK_sept2017/deepDNA/test/"
rootGenome = r"/data/tkj375/DNA_proj/from_AK_sept2017/deepDNA/test/"


rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
rootDevelopment = r"/data/tkj375/DNA_proj/development/"


############################################################
## Reading in results from frq model
############################################################

#On pc
rootFrq = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/Ecoli/"
rootFrq = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/CElegans/inclRepeats/"

#On binf
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/EColi/'

file = "frqModel_k4.file"
fileName = rootFrq + file

flankSize = 4
frqDict = dnaNet.getResultsFrqModel(fileName = fileName, flankSize = flankSize )


############################################################
## Running on specific genomes (in one go)  
############################################################

##########################
## MLP's
##########################

#EColi

#Here:
rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/EColi/"
fileName = "Ecoli_genome.fa"
fileGenome = rootGenome +fileName

nrEpochs = 100
stepsPerEpoch = 10
nrTrainSamples = 1000000
trainDataInterval = [0,2000000]
nrTestSamples = 200000
testDataInterval = [3000000,3400000]
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/EColi/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = [100],  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 0)

   

#On binf:
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/EColi/"
fileName = "Ecoli_genome.fa"
fileGenome = rootGenome +fileName

nrEpochs = 1000
stepsPerEpoch = 10
nrTrainSamples = 1000000
trainDataInterval = [0,2000000]
nrTestSamples = 200000
testDataInterval = [3000000,3400000]
nrHiddenUnits = [50,50]
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/EColi/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = nrHiddenUnits,  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 1)

#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/EColi/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, inclFrqModel_b = inclFrqModel_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/EColi/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = [100],  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 1)
 

#C Elegans:

#On binf:
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/CElegans/"
fileName = "ce6_ws190_chrom.fa"
fileGenome = rootGenome +fileName

nrEpochs = 100
stepsPerEpoch = 10
nrTrainSamples = 5000000
trainDataInterval = [0,6000000]
nrTestSamples = 1000000
testDataInterval = [6000000,10000000]
nrHiddenUnits = [50,50]
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/CElegans/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = nrHiddenUnits,  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 1)

#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/CElegans/inclRepeats/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, inclFrqModel_b = inclFrqModel_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/CElegans/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = [100],  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 1)
 


#Human
rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName

nrEpochs = 250
stepsPerEpoch = 100
batchSize = 1000
nrTrainSamples = 50000000
trainDataInterval = [0,110000000]
nrTestSamples = 10000000
testDataInterval = [110000000,130000000]
nrHiddenUnits = [200,100,50]

modelName = 'ownSamples/human/inclRepeats/modelMLP1'
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = nrHiddenUnits,  augmentWithRevComplementary_b = 0, batchSize = batchSize, shuffle_b = 0, on_binf_b = 1)

#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
file = "frqModel_chr10_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, inclFrqModel_b = inclFrqModel_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = 'ownSamples/CElegans/modelMLP_test', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = [100],  augmentWithRevComplementary_b = 0, batchSize = 50, shuffle_b = 0, on_binf_b = 1)


#Augmenting with the rev'ed complementaries:
augmentWithRevComplementary_b = 1
augmentTestDataWithRevComplementary_b = 0
modelName = 'ownSamples/human/inclRepeats/modelMLP1_augCompl_'
dnaNet.allInOneWithDynSampling_MLPmodel(learningRate = 0.1, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, nrHiddenUnits = nrHiddenUnits,  augmentWithRevComplementary_b = augmentWithRevComplementary_b, augmentTestDataWithRevComplementary_b = augmentTestDataWithRevComplementary_b,batchSize = batchSize, shuffle_b = 0, on_binf_b = 1)





flanksList = [10,20,30,40,50]
for i in range(5):
    flankSize = flanksList[i]
    modelName = 'ownSamples/human/inclRepeats/modelMLP_2layer_50units_flanks' + str(flankSize) + '_20mio'
    dnaNet.allInOne_MLPmodel(learningRate = 0.05, customFlankSize = flankSize, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = 20000000, trainDataInterval = [0,1e26] , nrValSamples = 4000000, valDataInterval = [0,1e26], nrTestSamples = 4000000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15','chr22'], nrHiddenUnits = [50,50],  augmentWithRevComplementary_b = 0, batchSize = 100, nrEpochs = 50, shuffle_b = 0, on_binf_b = 1)





##########################
## Conv's
##########################



# EColi ##########################

rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/EColi/"
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/EColi/"
fileName = "Ecoli_genome.fa"
fileGenome = rootGenome +fileName

#1D:
nrEpochs = 100
stepsPerEpoch = 10
batchSize = 1000
nrTrainSamples = 1000000
trainDataInterval = [0,2000000]
nrTestSamples = 200000
testDataInterval = [3000000,3400000]

learningRate = 0.05
poolAt = [1,3]
lengthWindows = [2, 3, 4, 4]
nrFilters = [4, 6, 8, 10] 

#One final dense layer:
hiddenUnits = [50] 

on_binf_b = 1
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, modelName = 'ownSamples/EColi/model1_onPC', nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/EColi/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
exclFrqModelFlanks_b = 0
modelName = 'ownSamples/EColi/model1WithFrqModel_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

#Larger w 2 final dense layers; the frq model output is fed into last of these:
nrFilters = [4, 6, 8, 10] 
hiddenUnits = [50, 20]  
modelName = 'ownSamples/EColi/model2Dyn_'
inclFrqModel_b = 0
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/EColi/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
excl FrqModelFlanks_b = 0
modelName = 'ownSamples/EColi/model2WithFrqModelDyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




# C Elegans ##########################

rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/CElegans/"
fileName = "ce6_ws190_chrom.fa"
fileGenome = rootGenome +fileName

nrEpochs = 1000
stepsPerEpoch = 10
batchSize = 1000
nrTrainSamples = 5000000
trainDataInterval = [0,10000000]
nrTestSamples = 2000000
testDataInterval = [10000000,20000000]

learningRate = 0.05
poolAt = [1,3]
lengthWindows = [2, 3, 4, 4]
nrFilters = [4, 6, 8, 10] 

#One final dense layer:
hiddenUnits = [50] 

on_binf_b = 1

modelName = 'ownSamples/CElegans/inclRepeats/model1Dyn'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Including results from frq model:
dynSamplesTransformStyle_b
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/CElegans/inclRepeats/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
exclFrqModelFlanks_b = 0
modelName = 'ownSamples/CElegans/inclRepeats/model1AndFrqDyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#Larger w 2 final dense layers; the frq model output is fed into last of these:
nrFilters = [8, 12, 16, 20] 
hiddenUnits = [50, 20]  
modelName = 'ownSamples/CElegans/inclRepeats/model2Dyn_'
inclFrqModel_b = 0
dropoutLastLayer_b = 0
dropoutVal = 0.25
momentum = 1e-6
learningRate = 0.01
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/CElegans/inclRepeats/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
exclFrqModelFlanks_b = 0
dropoutLastLayer_b = 0
dropoutVal = 0.25
momentum = 1e-6
learningRate = 0.01
modelName = 'ownSamples/CElegans/inclRepeats/model2AndFrqDyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#More shallow conv but with more filter; w 2 final dense layers; the frq model output is fed into last of these:
lengthWindows = [4, 6]
nrFilters = [20, 30] 
poolAt = [1]
hiddenUnits = [50, 20]  
modelName = 'ownSamples/CElegans/inclRepeats/model3Dyn_'
inclFrqModel_b = 0
flankSizeFrqModel = -1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/CElegans/inclRepeats/'
file = ""
frqModelFileName = rootFrq + file
exclFrqModelFlanks_b = 0
dropoutLastLayer_b = 0
dropoutVal = 0.25
momentum = 1e-3
learningRate = 0.02
on_binf_b = 1
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Including results from frq model:
inclFrqModel_b = 1
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/CElegans/inclRepeats/'
file = "frqModel_k4.file"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 4
exclFrqModelFlanks_b = 0
learningRate = 0.01
modelName = 'ownSamples/CElegans/inclRepeats/model2AndFrqDyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#  STILL WORKS?? Shuffling the outerparts of the flanks:
dnaNet.allInOne_ConvModel(genomeFileName = fileGenome, modelName = 'ownSamples/CElegans/inclRepeats/model1', nrTrainSamples = 1000000, trainDataInterval = [0,5000000] ,  nrValSamples = 400000, valDataInterval = [5000000,7000000], nrTestSamples = 400000, testDataInterval = [7000000,9000000], genSamplesFromRandomGenome_b = 0,  genSamples_b = 1, augmentWithRevComplementary_b = 0, batchSize = 128, nrEpochs = 100, shuffle_b = 1, inner_b = 0, shuffleLength = 45, on_binf_b = 1)





# fruit fly/melanogaster ##########################

rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/DMelanogaster/"
fileName = r"dmel-all-chromosome-r6.18.fasta"
fileGenome = rootGenome +fileName

nrEpochs = 100
stepsPerEpoch = 10
batchSize = 1000
nrTrainSamples = 1000000
trainDataInterval = [0,5000000]
nrTestSamples = 400000
testDataInterval = [5000000,7000000]

learningRate = 0.1
poolAt = [2,4]
lengthWindows = [2, 3, 4, 4, 5, 10]
nrFilters = [10, 20, 30, 40, 50, 60]   

on_binf_b = 1
modelName = 'ownSamples/fruit_fly/inclRepeats/model1Dyn'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, nrHiddenUnits = 50, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 






# Human genome ##########################


rootGenome = r"/Users/newUser/Documents/clouds/Sync/Bioinformatics/various_python/DNA_proj/data/human/"

rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName

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


#test model
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




#The model
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [3, 3,  6, 6, 9, 9]
nrFilters = [64, 64, 96, 96, 128, 128] 
padding = 'valid'
sizeOutput=4
# One final dense layers:
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

dnaNet.allInOneWithDynSampling_ConvModel_I(nrOuterLoops = nrOuterLoops, firstIterNr = 0, nrOfRepeats = nrOfRepeats, firstRepeatNr = 12, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval0 = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

dnaNet.allInOneWithDynSampling_ConvModel_I_testOnly(nrOuterLoops = 1, firstIterNr = 3 ,dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



dnaNet.allInOneWithDynSampling_ConvModel_II(dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, sizeOutput = sizeOutput, genomeFileName = fileGenome, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Larger w 2 final dense layers; the frq model output is fed into last of these:
nrEpochs = 100
nrFilters = [8, 12, 16, 20] 
hiddenUnits = [50, 20]  
flankSizeFrqModel = -1
exclFrqModelFlanks_b = 0
dropoutLastLayer_b = 0
dropoutVal = 0.25
momentum = 1e-3 #default
learningRate = 0.02

inclFrqModel_b = 0
modelName = 'ownSamples/human/inclRepeats/model2Dyn'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, modelIs1D_b = 1, genomeFileName = fileGenome, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  onlyOneRandomChromo_b = onlyOneRandomChromo_b, avoidChromo = avoidChromo, genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits,  nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


inclFrqModel_b = 1
modelName = 'ownSamples/human/inclRepeats/model2AndFrqDyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = 0, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Very large model (model that made the 47.5 % ... ?)

nrEpochs = 400
batchSize = 500
stepsPerEpoch = 200
nrTrainSamples = 200000000
trainDataInterval = [0,2000000000]
nrTestSamples = 40000000
testDataInterval = [2000000000,3000000000]

#for testing it/getting the duration of an epoch
nrEpochs = 3
batchSize = 500
stepsPerEpoch = 200
nrTrainSamples = 300000
trainDataInterval = [0,2000000]
nrTestSamples = 20000
testDataInterval = [2000000,3000000]


#model1
inclFrqModel_b = 1
str_inclFrqModel_b = '1'
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
lengthWindows = [3, 3]
nrFilters = [200, 200] 
# One final dense layers:
hiddenUnits = [50, 20] 
onlyOneRandomChromo_b = 0
#??
avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1
string = '_2layers_flat200_learningRate001'
modelName = 'ownSamples/human/inclRepeats/model1_InclFrq' + str_inclFrqModel_b + string + '_bigRun_'


#model2
inclFrqModel_b = 1
sizeOutput = 2
str_inclFrqModel_b = '1'
insertFrqModel_b = 1
str_insertFrqModel_b = '1'
frqSoftmaxed_b = 0
str_frqSoftmaxed_b = '0'
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
lengthWindows = [3, 3, 4, 4]
nrFilters = [20, 20, 20, 20] 
# One final dense layers:
hiddenUnits = [50, 20] 
string = '_4layers_20flat_learningRate001_dropOut01_'
modelName = 'ownSamples/human/inclRepeats/model2_I_InclFrq' + str_inclFrqModel_b + 'Inserted' + str_insertFrqModel_b + '_softmaxed' +str_frqSoftmaxed_b  + string + '_bigRun_'
modelFileName = modelName


#model3
learningRate = 0.001
pool_b = 0
poolAt = []
lengthWindows = [2, 3, 4, 4]
nrFilters = [16, 64, 32, 16] 
#Two final dense layers:
hiddenUnits = [20, 10] 
str = '4layers_varying16To16_learningRate01'
modelName = 'ownSamples/human/inclRepeats/model3Conv1dDyn_' + str + '_100_50_units2Hidden'  
modelDescr = str+'_100_50_units2Hidden'
modelName = 'ownSamples/human/inclRepeats/model3_ExclFrq_tesRun_'



#model4
learningRate = 0.001
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
lengthWindows = [2, 3, 4, 4]
nrFilters = [200, 200, 100, 100] 
#Two final dense layers:
hiddenUnits = [100, 100] 
str = '4layers_desc200To100_learningRate0001'
modelName = 'ownSamples/human/inclRepeats/model4Conv1dDyn_' + str + '_100_100_units2Hidden'  
modelDescr = str+'_100_100_units2Hidden'
modelName = 'ownSamples/human/inclRepeats/model4_ExclFrq_testRun_'


#model 5:
poolAt = [1,3,5]
lengthWindows = [2, 3, 4, 4, 6, 6]
nrFilters = [10, 20, 20, 30, 30, 40]
hiddenUnits = [50, 20] #model 5 uses this
str = '6layers_rising10to60_learningRate01'
modelName = 'ownSamples/human/inclRepeats/model5Conv1dDyn_' + str + '_50_20_units2Hidden'  
modelDescr = str+'_50_20_units2Hidden'
modelName = 'ownSamples/human/inclRepeats/model5_ExclFrq_bigRun_'


#model 6:
inclFrqModel_b = 0
sizeOutput = 2
str_inclFrqModel_b = '0'
frqSoftmaxed_b = 0
str_frqSoftmaxed_b = '0'
learningRate = 0.001
pool_b = 0
poolAt = [2, 4]
maxPooling_b = 1
lengthWindows = [2, 3, 4, 4, 5, 10]
nrFilters = [10, 20, 30, 40, 50, 60]
hiddenUnits = [50] #model 6 uses this
str = '6layers_rising10to60_learningRate001'
modelName = 'ownSamples/human/inclRepeats/model6Conv1dDyn_I_' + str + '_50_units1Hidden'  
modelFileName = modelName

#model 7:
inclFrqModel_b = 0
sizeOutput = 2
str_inclFrqModel_b = '0'
frqSoftmaxed_b = 0
str_frqSoftmaxed_b = '0'
learningRate = 0.001
pool_b = 0
poolAt = [2, 4]
maxPooling_b = 1
lengthWindows = [2, 3, 4, 4, 5, 10]
nrFilters = [30, 32, 34, 36, 38, 40]
hiddenUnits = [50] #model 6 uses this
str = '6layers_rising30to40_learningRate001'
modelName = 'ownSamples/human/inclRepeats/model7Conv1dDyn_I_ExclFrq_' + str + '_50_units1Hidden'  
modelFileName = modelName

#For Merging the trained model7 with the frq model:
inclFrqModel_b = 1
sizeOutput = 2
str_inclFrqModel_b = '0'
frqSoftmaxed_b = 0
str_frqSoftmaxed_b = '0'
learningRate = 0.001
str = '6layers_rising30to40_learningRate001'
modelName = 'ownSamples/human/inclRepeats/model7Conv1dDyn_I_ExclmergeIncl_' + str + '_50_units1Hidden'  
modelFileName = modelName

#Run it (any of the models):
dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
file = "frqModel_k5.txt"
frqModelFileName = rootFrq + file
flankSizeFrqModel = 5
exclFrqModelFlanks_b = 0
augmentWithRevComplementary_b = 0
dropoutLastLayer_b = 0
dropoutVal = 0.1
pool_b = pool_b
maxPooling_b = maxPooling_b
optimizer = 'ADAM'
momentum = 0.9 #default, but we use Adam here, so the value here isn't used
learningRate = learningRate
onlyOneRandomChromo_b = 0
avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'] 
on_binf_b = 1 
dnaNet.allInOneWithDynSampling_ConvModel_I(dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, optimizer = optimizer, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, frqSoftmaxed_b = frqSoftmaxed_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

dnaNet.allInOneWithDynSampling_ConvModel_II(dynSamplesTransformStyle_b = dynSamplesTransformStyle_b, optimizer = optimizer, learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, sizeOutput = sizeOutput, genomeFileName = fileGenome, frqSoftmaxed_b = frqSoftmaxed_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, modelFileName = modelFileName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



***************************************************************
** Run the prediction incl the frq model in version II
***************************************************************

#Read genome data:

#Ecoli
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/EColi/"
fileGenome = rootGenome + 'Ecoli_genome.fa' #ecoli

#Celegans
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/CElegans/"
fileName = "ce6_ws190_chrom.fa"
fileGenome = rootGenome +fileName

#Human
rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

genomeData = dnaNet.encodeGenome(fileGenome, outputAsDict_b = 0, startAtPosition = testDataInterval[0],
               endAtPosition = testDataInterval[0] + 10000000)

#genomeDataExcerpt = genomeData[0][:10000],genomeData[1][:10000]
avgPred = dnaNet.predictAcrossGenome_II(frqModelFileName = frqModelFileName, 
                        flankSizeFrqModel =  flankSizeFrqModel,
                        modelFileName = modelFileName, 
                        genomeData = genomeData ,
                        outputEncodedOneHot_b = 1,
                        outputEncodedInt_b = 0,
                        outputEncodedType = 'int8',
                        convertToPict_b = 0,
                        augmentWithRevComplementary_b = 0, 
                        customFlankSize_b = 0, 
                        customFlankSize = 50, 
                        shuffle_b = 0,
                        inner_b = 1, 
                        shuffleLength = 5,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        save_predPlot_b = 1,
                        predPlotName = 'type_II',
                        on_binf_b = 1)





#Augmenting with the rev'ed complementaries:
augmentWithRevComplementary_b = 1
augmentTestDataWithRevComplementary_b = 0
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 



#Another large model:

#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName


nrEpochs = 250
stepsPerEpoch = 100
batchSize = 1000
nrTrainSamples = 50000000
trainDataInterval = [0,110000000]
nrTestSamples = 10000000
testDataInterval = [110000000,130000000]

poolAt = [1,3]
lengthWindows = [2, 3, 4, 5, 10]
nrFilters = [50, 45, 40, 35, 30]
hiddenUnits = [50]
  
str = '5layers_desc50to30_learningRate01'
modelName = 'ownSamples/human/inclRepeats/model7Conv1dDyn_' + str + '_50_units1Hidden'  
modelDescr = str+'_50_units2Hidden'

inclFrqModel_b = 0
frqModelFileName = ""
flankSizeFrqModel = -1
exclFrqModelFlanks_b = 0
optimizer = 'SGD'
dropoutLastLayer_b = 0
dropoutVal = 0.1
momentum = 0.1 #default
learningRate = 0.1
augmentWithRevComplementary_b = 0
modelName = 'ownSamples/human/inclRepeats/model7Dyn_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 

#Augmenting with the rev'ed complementaries:
augmentWithRevComplementary_b = 1
augmentTestDataWithRevComplementary_b = 0
modelName = 'ownSamples/human/inclRepeats/model7Dyn_wAug_'
dnaNet.allInOneWithDynSampling_ConvModel(learningRate = learningRate, momentum = momentum,  modelIs1D_b = 1, genomeFileName = fileGenome, inclFrqModel_b = inclFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, nrTrainSamples = nrTrainSamples, trainDataInterval = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval, genSamplesFromRandomGenome_b = 0,  genSamples_b = 1,  lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  pool_b = 1, maxPooling_b = 1, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropoutLastLayer_b = dropoutLastLayer_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




#Several:

for i in range(0,1):
    if i ==0:
        str = '4layers_rising20to50_learningRate005'
        learningRate = 0.05
        poolAt = [1,3]
        lengthWindows = [2, 3, 4, 4]
        nrFilters = [20, 30, 40, 50]    
    elif i == 1:
        str = '4layers_rising20to50_learningRate01'
        learningRate = 0.1
        poolAt = [1,3]
        lengthWindows = [2, 3, 4, 4]
        nrFilters = [20, 30, 40, 50]   
    elif i == 2:
        str = '6layers_rising10to60_learningRate005'
        learningRate = 0.05
        poolAt = [2,4]
        lengthWindows = [2, 3, 4, 4, 5, 10]
        nrFilters = [10, 20, 30, 40, 50, 60]    
    elif i == 3:
        str = '6layers_rising10to60_learningRate01'
        learningRate = 0.1
        poolAt = [2,4]
        lengthWindows = [2, 3, 4, 4, 5, 10]
        nrFilters = [10, 20, 30, 40, 50, 60]           
    modelName = 'ownSamples/human/inclRepeats/modelConv1d_' + str + '_50units1Hidden_20mio'  
    modelDescr = str+'_50units1Hidden'
    dnaNet.allInOne_ConvModel(learningRate = learningRate, genomeFileName = fileGenome, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 20000000, trainDataInterval = [0,1e26] , nrValSamples = 4000000, valDataInterval = [0,1e26], nrTestSamples = 4000000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 50, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 25, shuffle_b = 0, on_binf_b = 1)


    dnaNet.allInOne_ConvModel(learningRate = learningRate, genomeFileName = fileGenome, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 200000, trainDataInterval = [0,1e6] , nrValSamples = 40000, valDataInterval = [0,1e6], nrTestSamples = 40000, testDataInterval = [0,1e6], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 50, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 1024, nrEpochs = 100, shuffle_b = 0, on_binf_b = 1)

str = '6layers_rising10to60_learningRate01'
learningRate = 0.1
poolAt = [2,4]
lengthWindows = [2, 3, 4, 4, 5, 10]
nrFilters = [10, 20, 30, 40, 50, 60]
modelName = 'ownSamples/human/inclRepeats/modelConv1d_' + str + '_50units1Hidden_20mio'  
modelDescr = str+'_50units1Hidden'
dnaNet.allInOne_ConvModel(learningRate = learningRate, genomeFileName = fileGenome, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 20000000, trainDataInterval = [0,1e26] , nrValSamples = 4000000, valDataInterval = [0,1e26], nrTestSamples = 4000000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 50, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 25, shuffle_b = 0, on_binf_b = 1)



#exp with more filters:
str = '4layers_rising40to100_learningRate01'
learningRate = 0.1
poolAt = [1,3]
lengthWindows = [2, 4, 6, 8]
nrFilters = [40, 60, 80, 100] 
modelName = 'ownSamples/human/inclRepeats/modelConv1d_' + str + '_30units1Hidden_20mio'  
modelDescr = str+'_30units1Hidden'
dnaNet.allInOne_ConvModel(bigLoopsNr = 2, startFrom = 1, loadModelFromFile_b = 1, learningRate = learningRate, genomeFileName = fileGenome, customFlankSize = 50, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 20000000, trainDataInterval = [0,1e26] , nrValSamples = 4000000, valDataInterval = [0,1e26], nrTestSamples = 4000000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 30, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 10, shuffle_b = 0, on_binf_b = 1)
#nr of params:
dnaNet.nrOfParametersConv(nrFilters, lengthWindows, 4)
409920
 
#exp with more filters:
str = '6layers_rising20to120_learningRate01'
learningRate = 0.1
poolAt = [1,3,5]
lengthWindows = [2, 4, 6, 8, 10, 12]
nrFilters = [20, 40, 60, 80, 100, 120] 
modelName = 'ownSamples/human/inclRepeats/modelConv1d_' + str + '_30units1Hidden_20mio'  
modelDescr = str+'_30units1Hidden'
dnaNet.allInOne_ConvModel(bigLoopsNr = 5, startFrom = 0, loadModelFromFile_b = 0, learningRate = learningRate, genomeFileName = fileGenome, customFlankSize = 50, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 20000000, trainDataInterval = [0,1e26] , nrValSamples = 4000000, valDataInterval = [0,1e26], nrTestSamples = 4000000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 30, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 2, shuffle_b = 0, on_binf_b = 1)
#nr of params
dnaNet.nrOfParametersConv(nrFilters, lengthWindows, 4)
1120160
   
   
#exp with more filters, larger flanks:
str = 'flanks100_3layers_rising40to80_learningRate01'
learningRate = 0.1
poolAt = [1,2]
lengthWindows = [2, 5, 10]
nrFilters = [40, 60, 80] 
modelName = 'ownSamples/human/inclRepeats/modelConv1d_' + str + '_30units1Hidden_2mio'  
modelDescr = str+'_30units1Hidden'
dnaNet.allInOne_ConvModel(bigLoopsNr = 10, startFrom = 0, loadModelFromFile_b = 0, learningRate = learningRate, genomeFileName = fileGenome, customFlankSize = 100, modelName = modelName, modelDescription = modelDescr, nrTrainSamples = 2000000, trainDataInterval = [0,1e26] , nrValSamples = 400000, valDataInterval = [0,1e26], nrTestSamples = 400000, testDataInterval = [0,1e26], genSamplesFromRandomGenome_b = 0, genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], lengthWindows = lengthWindows, nrHiddenUnits = 30, nrFilters = nrFilters, augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 2, shuffle_b = 0, on_binf_b = 1)
#nr of params:
dnaNet.nrOfParametersConv(nrFilters, lengthWindows, 4)
409920
 
##########################
## LSTM's ... and more 
##########################





#Write exonic info to file:
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName
inputFilename = rootGenome + 'exonInfoHg19'
outputFilename = rootGenome + 'exonInfoHg19Binary_CDSonly.txt' 
d = dnaNet.generateExonicInfoFromFile(inputFilename = inputFilename, genomeFilename = fileGenome, chromoNameBound = 10, outputFilename = outputFilename)
#look at results:
exonicInfoBinaryFileName = outputFilename
startAtPosition = 0
endAtPosition = 1000000
outputGenomeString_b = 1
XencDict, XencRepeatDict, XencExonicDict, XchrDict, XchrNsDict = dnaNet.encodeGenome(fileName = fileGenome, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = outputGenomeString_b, outputEncoded_b = 1, outputEncodedOneHot_b = 1, outputEncodedInt_b = 0, outputAsDict_b = 1)

rep,exo = dnaNet.genomeStats(XencDict, XencRepeatDict, XencExonicDict)


#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName


#rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName


#for looking at the flow:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 1
firstRepeatNr = 116 #loads in model from repeatNr 115!
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 1
batchSize = 10
stepsPerEpoch = 10
trainDataIntervalStepSize = 200
trainDataInterval = [20000,21000]
nrTestSamples = 10
testDataInterval = [21000,22000]


#for testing:
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


nrEpochs = 450
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 200000000
trainDataInterval = [0,400000000]
nrTestSamples = 20000000
testDataInterval = [400000000,600000000]


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



dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
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


labelsCodetype = 2 #1: ERO -- ExonicRepeatOther prediction
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary.txt'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, convLayers_b = 0, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,   lengthWindows = lengthWindows, hiddenUnits = hiddenUnits, nrFilters = nrFilters,  padding = padding, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = 1, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Repeat-Other (RO) prediction
labelsCodetype = 3 #1: RO -- RepeatOther prediction
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary_CDSonly.txt'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Conv-LSTM fused with RO-model:
labelsCodetype = 0
fusedWitEROmodel_b = 1
eroModelFileName = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/human/inclRepeats/modelLSTM__1Conv2LayerLstm_forRO_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr50'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, fusedWitEROmodel_b = fusedWitEROmodel_b, eroModelFileName =eroModelFileName, convLayers_b = 1, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




#Exonic-Repeat-Other (ERO) prediction
labelsCodetype = 2 #1: ERO -- ExonicRepeatOther prediction
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary_CDSonly.txt'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, convLayers_b = 1, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Conv-LSTM fused with ERO-model:
labelsCodetype = 0
fusedWitEROmodel_b = 1
eroModelFileName = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/human/inclRepeats/modelLSTM__1Conv2LayerLstm_forERO_CDSonly_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr37'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, fusedWitEROmodel_b = fusedWitEROmodel_b, eroModelFileName =eroModelFileName, convLayers_b = 1, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 


#Conv-LSTM only on repeats:
labelsCodetype = 0
getOnlyRepeats_b = 1
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary_CDSonly.txt'
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, getOnlyRepeats_b = getOnlyRepeats_b , nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, fusedWitEROmodel_b = 0, eroModelFileName ='', convLayers_b = 1, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 




#Only conv:

#single chromo
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19_chr10.fa"
fileGenome = rootGenome +fileName

#rootGenome = r"/isdata/kroghgrp/krogh/scratch/db/hg19/"
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
fileName = r"hg19.fa"
fileGenome = rootGenome +fileName

#for testing:
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 2
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
nrOfRepeats = 100
firstRepeatNr = 0
testDataIntervalIdTotrainDataInterval_b = 1
nrEpochs = 100
batchSize = 500
stepsPerEpoch = 100
trainDataIntervalStepSize = 0 #3000000000
trainDataInterval = [0,1000000000]
nrTestSamples = 1000000
testDataInterval = [10000000,12000000]

exonicInfoBinaryFileName  = ''
inclFrqModel_b = 0
insertFrqModel_b = 0
customFlankSize = 50
overlap = 0
pool_b = 0
poolAt = [1, 3]
maxPooling_b = 0
poolStrides = 1
lengthWindows = [2,3,4,6,8,10]
nrFilters = [200, 200, 200, 200, 200, 200] 
filterStride = 1
#parallel LSTMs:
nrOfParallelLSTMstacks = 0
#Final dense layers:
finalDenseLayers_b = 1
hiddenUnits = [100, 50]
#Nr of lstm layers:
nrLSTMlayers = 0
padding = 'valid'

dynSamplesTransformStyle_b = 0
inclFrqModel_b = inclFrqModel_b
insertFrqModel_b = insertFrqModel_b
rootFrq = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'
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

learningRate = 0.001

subStr= '_TEST_6ConvLayers2Dense_200flat_learningRate001_dropOut00'
modelName = 'ownSamples/human/inclRepeats/modelLSTM_' + subStr
modelDescr = subStr


onHecaton_b = 0
onlyConv_b = 1
leftRight_b = 0
labelsCodetype = 0
getOnlyRepeats_b = 0
exonicInfoBinaryFileName  = ''
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(onHecaton_b = onHecaton_b, labelsCodetype = labelsCodetype, onlyConv_b = onlyConv_b, leftRight_b = leftRight_b, getOnlyRepeats_b = getOnlyRepeats_b , nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, fusedWitEROmodel_b = 0, eroModelFileName ='', convLayers_b = 1, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b) 









#Look at predictions:

#Test of a model only on repeat-positions:
labelsCodetype = 0
getOnlyRepeats_b = 1
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary_CDSonly.txt'
testOnly_b = 1
subStr = '_1LayerConv2LayerLstm1LayerDense20_flanks50_win4_stride1_overlap0_dropout00'
modelName = 'ownSamples/human/inclRepeats/modelLSTM_' + subStr
nrOuterLoops = 1
firstIterNr = 0
nrOfRepeats = 150
firstRepeatNr = 149
dnaNet.allInOneWithDynSampling_ConvLSTMmodel(labelsCodetype = labelsCodetype, getOnlyRepeats_b = getOnlyRepeats_b , nrOuterLoops = nrOuterLoops, firstIterNr = firstIterNr, nrOfRepeats = nrOfRepeats,  firstRepeatNr = firstRepeatNr, fusedWitEROmodel_b = 0, eroModelFileName ='', convLayers_b = 1, nrLSTMlayers = nrLSTMlayers, overlap = overlap, learningRate = learningRate, momentum = momentum,  genomeFileName = fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, customFlankSize = customFlankSize, inclFrqModel_b = inclFrqModel_b, insertFrqModel_b = insertFrqModel_b, exclFrqModelFlanks_b = exclFrqModelFlanks_b, frqModelFileName = frqModelFileName, flankSizeFrqModel = flankSizeFrqModel, modelName = modelName, testDataIntervalIdTotrainDataInterval_b = testDataIntervalIdTotrainDataInterval_b, trainDataIntervalStepSize = trainDataIntervalStepSize, trainDataInterval0 = trainDataInterval , nrTestSamples = nrTestSamples, testDataInterval = testDataInterval,   genSamples_b = 1,  nrOfParallelLSTMstacks = nrOfParallelLSTMstacks, lengthWindows = lengthWindows, finalDenseLayers_b = finalDenseLayers_b, hiddenUnits = hiddenUnits, nrFilters = nrFilters, padding = padding, filterStride = filterStride, pool_b = pool_b, maxPooling_b = maxPooling_b, poolAt = poolAt, poolStrides = poolStrides, optimizer = optimizer, dropoutVal = dropoutVal, dropout_b = dropout_b, augmentWithRevComplementary_b = augmentWithRevComplementary_b, batchSize = batchSize, nrEpochs = nrEpochs, stepsPerEpoch = stepsPerEpoch, shuffle_b = 0, on_binf_b = on_binf_b, testOnly_b = testOnly_b) 





#Cut down the length of the seq's read in:
exonicInfoBinaryFileName  = r'/isdata/kroghgrp/tkj375/data/DNA/human/hg19/exonInfoHg19Binary.txt'
genomeDict = dnaNet.encodeGenome(fileGenome, exonicInfoBinaryFileName = exonicInfoBinaryFileName, outputAsDict_b = 0, startAtPosition = 1000000, endAtPosition = 2000000)


labelsCodetype = 2
fileNameModel = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/human/inclRepeats/modelLSTM__1Conv2LayerLstm_forERO_flanks50_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr56'
outputWig_b = 0
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, batchSize = 1024, windowLength = 10000, stepSize = 2500, wigFileName = 'ownSamples/CElegans/inclRepeats/model3_1mBound', on_binf_b = 1, shuffle_b = 0)







#LSTM
dnaNet.allInOne_LSTMmodel(genomeFileName = fileGenome,  outputEncodedOneHot_b  = 1, outputEncodedInt_b = 0, letterShape = 4, modelName = modelName, customFlankSize = 10,  nrTrainSamples = 5000000, trainDataInterval = [0,1e26] , nrValSamples = 2000000, valDataInterval = [0,1e26], nrTestSamples = 2000000, testDataInterval = [0,1e26], genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], augmentWithRevComplementary_b = 0, batchSize = 100, nrEpochs = 2, shuffle_b = 0, on_binf_b = 1, modelDescription = modelDescr)


#conv1D+ LSTM
str = '_conv1DLSTM1layer_dense'
learningRate = 0.05
poolAt = [1,3]
modelName = 'ownSamples/human/inclRepeats/modelconv1DLSTM_' + str + '_3layerConv1D1layerLSTMBidir50_5mio_50'  
modelDescr = str + '_3layerConv1D1layerLSTMBidir50_5mio_50'  
dnaNet.allInOne_LSTMmodel(learningRate = learningRate, genomeFileName = fileGenome, outputEncodedOneHot_b  = 1, outputEncodedInt_b = 0, letterShape = 4, modelName = modelName, customFlankSize = 50,  nrTrainSamples = 5000000, trainDataInterval = [0,1e26] , nrValSamples = 2000000, valDataInterval = [0,1e26], nrTestSamples = 2000000, testDataInterval = [0,1e26], genSamples_b = 1, onlyOneRandomChromo_b = 1, avoidChromo = ['chrX', 'chrY', 'chrM', 'chr15', 'chr22'], augmentWithRevComplementary_b = 0, pool_b = 1, maxPooling_b = 1, poolAt = poolAt, batchSize = 528, nrEpochs = 2, shuffle_b = 0, on_binf_b = 1, modelDescription = modelDescr)




############################################################
## Running on randomly generated samples/genome  
############################################################

#In one go:
dnaNet.allInOne(trainDataInterval = [0,1000] , valDataInterval = [0,400], testDataInterval = [0,400], genSamplesFromRandomGenome_b = 0, randomGenomeSize = 10000, randomGenomeFileName = 'rndGenome.txt', genSamples_b = 0, augmentWithRevComplementary_b = 0, batchSize = 64, nrEpochs = 10, shuffle_b = 0, on_binf_b = 0)


#Random genome and random samples:

#file name for genome file:
fileName = root + 'rndGenome.txt'

#Generate random genome (will write it to the file above)
lenGenome = 5000000 #number of bases
dnaNet.genRandomGenome(length = lenGenome, fileName = fileName )

#Draw samples from the randon genome:
nrSamples = 1000000
X,Y = dnaNet.genSamples_I(nrSamples = nrSamples, fromGenome_b = 1, genomeFileName = fileName, getFrq_b =1, augmentWithRevComplementary_b = 0)



############################################################

Predictions

############################################################

import dnaConNetL as dnaNet

root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"


rootGenome = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/from_AK_sept2017/deepDNA/test/"
rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/"
rootGenome = r"/data/tkj375/DNA_proj/from_AK_sept2017/deepDNA/test/"
#C Elegans
rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/CElegans/"
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/CElegans/"
fileName = "ce6_ws190_chrom.fa"
#fruit fly/melanogaster
rootGenome = r"/data/tkj375/DNA_proj/Inputs/genomeData/fruit_fly/"
fileName = r"dmel-all-chromosome-r6.18.fasta"
#S Cerevisiae:
rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/SCerevisiae/"
rootGenome = r"/data/tkj375/DNA_proj/Inputs/genomeData/SCerevisiae/"
fileName = "GCA_000146045.2_R64_genomic"
#E Coli
rootGenome = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/genomeData/EColi/"
rootGenome = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/genomeData/EColi/"
fileName = "Ecoli_genome.fa"

fileGenome = rootGenome +fileName

rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"

#Load saved model from:
rootModel = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Input/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/EColi/"

#Read genome data:
fileGenome = rootGenome + 'Ecoli_genome.fa' #ecoli
fileGenome = rootGenome + 'ce6_ws190_chrom.fa' #c elegans
fileGenome = rootGenome +  "GCA_000146045.2_R64_genomic.fna" #S Cervisiae


genomeData = dnaNet.encodeGenome(fileGenome, outputAsDict_b = 0)
#or as dict:
genomeDict = dnaNet.encodeGenome(fileGenome, outputAsDict_b = 1)
#Cut down the length of the seq's read in:
genomeDict = dnaNet.encodeGenome(fileGenome, outputAsDict_b = 1, endAtPosition = 1000000)


#For Ecoli (and others) there's no chromo's so the chromoName will just be the
#first fasta line starting with '>' and it may be much longer than a few char's:
genomeDict = dnaNet.encodeGenome(fileGenome, outputAsDict_b = 1, chromoNameBound = 1000)

#Get test data
Xt, Yt, Rt = dnaNet.getAllSamplesFromGenome(genomeData = genomeData, flankSize = 50, augmentWithRevComplementary_b = 0)

#Run prediction
rootOutput = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_nets/ownSamples/EColi/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/EColi/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/theanoDNA_proj/results_nets/ownSamples/CElegans/inclRepeats/"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ownSamples/DMelanogaster/inclRepeats/"



#EColi
modelName = 'model3'
fileNameModel = rootOutput + modelName
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, batchSize = 128, windowLength = 10,  stepSize = 5, wigFileName = 'ownSamples/EColi/model3', on_binf_b = 0, shuffle_b = 0)

dnaNet.predictAcrossGenome(fileNameModel, genomeData, windowLength = 1000)

#CElegans
modelName = 'model3'
fileNameModel = rootOutput + modelName
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, batchSize = 1024, windowLength = 10000, stepSize = 2500, wigFileName = 'ownSamples/CElegans/inclRepeats/model3_1mBound', on_binf_b = 1, shuffle_b = 0)

#Shuffling the input:
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, windowLength = 100, wigFileName = 'ownSamples/CElegans/inclRepeats/model2_shuffledTest', on_binf_b = 1, shuffle_b = 1, inner_b = 2, shuffleLength = 5)
#Model trained on shuffled data, tested here on unshuffled data:
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, batchSize = 1024, windowLength = 1000, wigFileName = 'ownSamples/CElegans/inclRepeats/model3', on_binf_b = 1, shuffle_b = 0)

#Droso
avgPred = dnaNet.predictAcrossGenomeDict(fileNameModel, genomeDict, windowLength = 1000, wigFileName = 'ownSamples/DMelanogaster/inclRepeats/model1', on_binf_b = 1)







'''

#from __future__ import print_function

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


###################################################################
## Reading in results from frq model while converting to one-hot encoding
###################################################################
        
def getResultsFrqModel(fileName, flankSize = 4, applySoftmax_b = 0):
    '''Fetches results from frq model (as provided in the file) and 
    outputs these as a dictionary mapping each word (: read from 
    beginning of left-hand flank to end of right hand flank, while
    skipping the position in the middle) to the frq of the four
    letters in the order of the one-hot encoding.
    
    '''
    
    #get results from frq model:
    resDict = frqM.readResults(fileName)
    
    outDict = {}
    
        
    if applySoftmax_b == 0:
        for word in resDict:
            
            newWord = word[:flankSize] + word[(flankSize+1):]
            value = resDict[word][3:] #the prob's are in col 4 -7 in the output from AK's kmerPredict pgm
            
            outDict[newWord] = value
            
    elif applySoftmax_b != 0:
        
        for word in resDict:
            
            newWord = word[:flankSize] + word[(flankSize+1):]
            value = resDict[word][3:] #the prob's are in col 4 -7 in the output from AK's kmerPredict pgm
            
            #Apply the "softmax": replace the prob-values with a length 4 vector having 1 at the highest
            #prob, and zeros on the remianing three:
            maxAt = np.argmax(np.asarray(value))
            newValue = alphabetOnehot[maxAt]
            
            outDict[newWord] = newValue
        

    return outDict
    


###############################################################################
##    Prediction
###############################################################################


def predictAcrossGenome(modelFileName, 
                        genomeData,
                        labelsCodetype = 0,
                        outputEncodedOneHot_b = 1,
                        outputEncodedInt_b = 0,
                        outputEncodedType = 'int8',
                        convertToPict_b = 0,
                        augmentWithRevComplementary_b = 0, 
                        customFlankSize_b = 0, 
                        customFlankSize = 50, 
                        shuffle_b = 0,
                        inner_b = 1, 
                        shuffleLength = 5,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence (encoded, as
    read in by readGenome) by using the input estimated/trained model.
    Input:
    
    genomeData: tuple dna-sequence, repeat info as returned be readGenome
    '''
    
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


    net = model_from_json(open(modelFileName).read())
    net.load_weights(modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
    sizeOutput = net.output_shape[1]

    genomeSeq, repeatInfoSeq, exonicInfoSeq =  genomeData

    lGenome = len(genomeSeq)
    
    print("Genome length: %d" % lGenome)

    #Read in the test data.
    #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
    Xt, Yt, Rt = getAllSamplesFromGenome(genomeData = genomeData, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)


#    root = r"/data/tkj375/DNA_proj/Inputs/"
#    root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
#    
#    tname= root + r"training.dat" #root + r"test.dat"
#    testDataInterval = [1000000, 2000000] 
#    testDataInterval = [0, 2000] 
#    Xt,Yt = getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
#  
#    print Xt.shape
#    print Yt.shape

#    Xt = Xt.astype('float32')
#    Yt = Yt.astype('float32')

    lSamples = len(Yt)
#
#    chunkSize = 10000
#    chunkNrSamples = chunkSize/(2*customFlankSize)
#    
#    nrOfPasses = lSamples/chunkSize
#
#    for i in range(nrOfPasses):
#        
#        for j in range(i*chunkSize, (i+1)*chunkSize):

    
    #Call the prediction
    pred = net.predict(Xt, batch_size = batchSize)
    
    print("Nr of sites: %d ; of which are predicted: %d" % (lSamples, pred.shape[0]))

    nrSteps = int((lSamples - windowLength)/stepSize)

    print("nrSteps ", nrSteps )
    
    cntCorr = 0.0
    cntCorrRep = 0.0
#    cntCorrA = 0.0 
#    cntCorrT = 0.0
#    cntCorrC = 0.0
#    cntCorrG = 0.0
    
#    cntA = 0
#    cntT = 0
#    cntC = 0
#    cntG = 0
    
    windowList = []
    windowArray = np.zeros(shape = windowLength, dtype = 'float32')
    avgPred = np.zeros(shape = nrSteps, dtype = 'float32')

    for j in range(nrSteps):
    
        #first window: read in the following windoLength worth of sites:
        if j == 0:
            
            for i in range(windowLength): #range(pred.shape[0]):
                
#                print " ".join(map(str,Yt[i])), " ".join(map(str,pred[i]))
            
                predIdx = np.argmax(pred[i])
                
#                print predIdx
                
                if Yt[i][predIdx] > 0.5:
                    cntCorr += 1.0
                    if Rt[i] > 0.5:
                        cntCorrRep += 1.0
                    windowList.append(1.0)
                    
                else:
                    windowList.append(0.0)
            
            
            windowArray = np.asarray(windowList)
            avgPred[j] = np.mean(windowArray) #and j = 0
            
        else:
                        
            i = j + windowLength
            predIdx = np.argmax(pred[i])
            
            #remove first elt from list
            windowList.pop(0)  
            #and append the next:
            if Yt[i][predIdx] > 0.5:
                cntCorr += 1
                if Rt[i] > 0.5:
                    cntCorrRep += 1.0
                windowList.append(1.0)
                    
            else:
                windowList.append(0.0)

            windowArray = np.asarray(windowList)
            avgPred[j] = np.mean(windowArray)
            
            
    plt.figure()       
    plt.plot(avgPred) 
    plt.savefig(modelFileName + '_predPlot.pdf' )    
    
    
    avgCorr = cntCorr/cntTot
    print("Chromo: %s , average prediction acc : %f" % (chromo, avgCorr))

    nrReps = np.sum(Rt)
    if nrReps > 0.5: #ie if there are any repeats recorded
        avgCorrRep = cntCorrRep/cntTotRep
        avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
        print("Chromo: %s, average prediction acc at repeats: %f and elsewhere: %f" % (chromo, avgCorrRep, avgCorrNonRep))
    else:
        print("Chromo: %s, no repeat sections were recorded in the genome data." % chromo)
            
            
def predictAcrossGenomeDict(modelFileName, 
                        genomeDictTuple,
                        labelsCodetype = 0,
                        outputEncodedOneHot_b = 1,
                        outputEncodedInt_b = 0,
                        outputEncodedType = 'int8',
                        convertToPict_b = 0,
                        augmentWithRevComplementary_b = 0, 
                        customFlankSize_b = 0, 
                        customFlankSize = 50, 
                        shuffle_b = 0,
                        inner_b = 1, 
                        shuffleLength = 5,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 50, 
                        Fourier_b = 0,
                        outputWig_b = 0,
                        wigFileName = '',
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence (encoded, as
    read in by readGenome) by using the input estimated/trained model.
    Input: 
    genomeDictTuple: structure genome seq dict, repeat info dict as output by readGenome
    (genome DNA-sequence/repeat info structured in dictionary mapping each chromosome to 
    its sequence/repeat info).'''
    
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"D:/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


    net = model_from_json(open(modelFileName).read())
    net.load_weights(modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
    sizeOutput = net.output_shape[1]

    cntCorrAll = 0.0
    cntTotAll = 0.0
    nrStepsAll = 0
    cntCorrRepAll = 0.0
    cntTotRepAll = 0.0
    nrRepsAll = 0
    
    chromoCnt = 0
    
    genomeDict, repeatInfoDict, exonicInfoDict = genomeDictTuple
    
    for chromo in genomeDict.keys():
        
        chromoCnt += 1

        genomeData = genomeDict[chromo], repeatInfoDict[chromo]

        lGenome = len(genomeData[0])
        
        print("Chromo: %s genome length: %d" % (chromo,lGenome))
    
        #Read in the test data.
        #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
        Xt, Yt, Rt = getAllSamplesFromGenome(genomeData = genomeData, labelsCodetype = labelsCodetype, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
    
    
    #    root = r"/data/tkj375/DNA_proj/Inputs/"
    #    root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
    #    
    #    tname= root + r"training.dat" #root + r"test.dat"
    #    testDataInterval = [1000000, 2000000] 
    #    testDataInterval = [0, 2000] 
    #    Xt,Yt = getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
    #  
        print("Shape flanks data across this chromo: ", Xt.shape)
        print("Shape midpoints data across this chromo: ", Yt.shape)
    
    #    Xt = Xt.astype('float32')
    #    Yt = Yt.astype('float32')
    
        lSamples = len(Yt)
    #
    #    chunkSize = 10000
    #    chunkNrSamples = chunkSize/(2*customFlankSize)
    #    
    #    nrOfPasses = lSamples/chunkSize
    #
    #    for i in range(nrOfPasses):
    #        
    #        for j in range(i*chunkSize, (i+1)*chunkSize):
    
        #Call the prediction
        pred = net.predict(Xt, batch_size = batchSize)
        
        print("Nr of sites: %d ; of which are predicted: %d" % (lSamples, pred.shape[0]))
    
        nrSteps = int((lSamples - windowLength)/stepSize)
    
        print("nrSteps ", nrSteps )
        
    #    print "nrSteps ", nrSteps 
        
        cntCorr = 0.0
        cntTot = 0.0
        cntCorrRep = 0.0
        cntTotRep = 0.0
#        cntCorrA = 0.0
#        cntCorrT = 0.0
#        cntCorrC = 0.0
#        cntCorrG = 0.0
#        
#        cntA = 0
#        cntT = 0
#        cntC = 0 
#        cntG = 0
        
        windowList = []
        windowArray = np.zeros(shape = windowLength, dtype = 'float32')
        avgPred = np.zeros(shape = nrSteps+1, dtype = 'float32')
    
        for j in range(nrSteps):
        
            #first window: read in the following windoLength worth of sites:
            if j == 0:
                
                for i in range(windowLength): #range(pred.shape[0]):
                    
    #                print " ".join(map(str,Yt[i])), " ".join(map(str,pred[i]))
                
                    predIdx = np.argmax(pred[i])
                    
    #                print predIdx
                    
                    if Yt[i][predIdx] > 0.5:
                        cntCorr += 1.0
                        if Rt[i] > 0.5:
                            cntCorrRep += 1.0
                        windowList.append(1.0)
                        
                    else:
                        windowList.append(0.0)
                        
                    cntTot += 1
                
                
                windowArray = np.asarray(windowList)
                avgPred[j] = np.mean(windowArray) #and j = 0
                
            else:
                
                #remove first stepSize elt's from list
                for k in range(stepSize):
                    windowList.pop(0)
                    
                #Append the windowLength - stepSize next elts:
                for l in range(windowLength - stepSize): 
                    i = j*stepSize + l
                    predIdx = np.argmax(pred[i])
                    if Yt[i][predIdx] > 0.5:
                            cntCorr += 1
                            if Rt[i] > 0.5:
                                cntCorrRep += 1.0
                                cntTotRep  += 1.0
                            windowList.append(1.0)
                            
                    else:
                        windowList.append(0.0)
                        if Rt[i] > 0.5:
                            cntTotRep  += 1.0
                    
                    cntTot += 1
    
                windowArray = np.asarray(windowList)
                avgPred[j] = np.mean(windowArray)
                
        #forcing the avgPred to be periodic (enabling the Fourier transform):
        avgPred[stepSize] = avgPred[0]
        print("Avg pred at 0: %f  and at nrSteps: %f" %(avgPred[0], avgPred[stepSize]) )
        
        plt.figure()
        plt.title(chromo + ' avg prediction')        
        plt.plot(avgPred) 
        plt.savefig(modelFileName + '_predPlot_' + chromo[30:70] + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )

        if Fourier_b == 1:
            
            #Fourier transform it:
            fftAvgPred = fft(avgPred) #scipy fast Fourier transform
            print( "Post fft: Avg pred at 0: %f  and at nrSteps: %f" %(avgPred[0], avgPred[stepSize]) )
            plt.figure()
            plt.title(chromo + ', fft avg prediction, excerpt')  
            start = int(nrSteps/34)
            end = int(nrSteps/33)
            plt.bar(range(start,end),fftAvgPred[start:end]) 
            plt.savefig(modelFileName + '_FourierTransformPredPlotZoom_2_' + chromo[30:70] + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
            
        
        avgCorr = cntCorr/cntTot
        print("Chromo: %s , average prediction acc : %f" % (chromo, avgCorr))

        nrReps = np.sum(Rt)
        if nrReps > 0.5: #ie if there are any repeats recorded
            avgCorrRep = cntCorrRep/cntTotRep
            avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
            print("Chromo: %s, average prediction acc at repeats: %f and elsewhere: %f" % (chromo, avgCorrRep, avgCorrNonRep))
        else:
            print("Chromo: %s, no repeat sections were recorded in the genome data." % chromo)

        cntCorrAll += cntCorr
        cntTotAll += cntTot
        nrStepsAll += nrSteps
        cntCorrRepAll += cntCorrRep
        cntTotRepAll += cntTotRep
        nrRepsAll += nrReps
        
        #if desired write to wig-file (for custom track in UCSC genome browser);
        #there will be one file/track per chromosome 
        if outputWig_b == 1:
                
            if wigFileName == '':
                useWigFileName = modelFileName
            try:    
                useWigFileName = rootOutput + wigFileName  + '_pred_' + chromo + '_win' + str(windowLength) + '_step' + str(stepSize) + '.wig'
                #flush the wig-file by opening it and closing it again:
                wigFile = open(useWigFileName, 'w')
            except IOError: #file name may be prohibited; so try cutting down the chromo's name 
                useWigFileName = rootOutput + wigFileName  + '_pred_chromoNN' + str(chromoCnt) + '.wig'
                #flush the wig-file by opening it and closing it again:
                wigFile = open(useWigFileName, 'w')
            wigFile.close()
            wigFile = open(useWigFileName, 'a')
            print("For chromo: %s will write to wig-file: %s" % (chromo, useWigFileName )  )
        
        
            s = 'browser position ' + chromo + ':1-' + str(lSamples) + "\n"
            wigFile.write(s)
            s = 'browser hide all' + "\n"
            wigFile.write(s)
            #	200 base wide points graph at every 300 bases, 50 pixel high graph
            #	autoScale off and viewing range set to [0:1000]
            #	priority = 20 positions this as the second graph
            #	Note, one-relative coordinate system in use for this format
            s = 'track type=wiggle_0 name="fixedStep" description="fixedStep format" visibility=full autoScale=off viewLimits=0:1000 color=0,200,100 maxHeightPixels=100:50:20 graphType=points priority=20' + "\n"
            wigFile.write(s)
            s='fixedStep chrom=' + chromo + ' start=1' + ' step=' + str(stepSize) + "\n"
            wigFile.write(s)
            #loop across the range and write to file in chunks of e.g. 1000 lines
            #first and last windowLength worth of positions are left at 0 (there are no
            #predictions there):
            s = ''
            for i in range(windowLength):
                pos = i + 1
                if pos%stepSize == 0:
                    s += str(0.0) + "\n"
#                s += str(pos) + '  ' + str(0.0) + "\n"
            wigFile.write(s)
            s = ''
            posStart = pos
#            sumAvgPred = 0
            for j in range(nrSteps): #nrSteps
#                print avgPred[j]
                pos = posStart + 1 + j*stepSize
#                sumAvgPred += avgPred[j]
                s += str(round(avgPred[j], 6)) + "\n"
#                if j%100 == 0:
#                    avg = sumAvgPred/100
##                    s += str(round(avg, 6)) + "\n"
#                    sumAvgPred = 0
##                    s += str(pos) + '  ' + str(round(avg, 6)) + "\n" #for variableStep wig-type
                if j%100  == 0 and j > 0:
                    wigFile.write(s)
                    s = '' #reset
            #in case nrSteps is not divisible by 100, we need to write this last piece to file too:
            if j%100 != 0:
                wigFile.write(s)
            s = ''
            posStart = pos
            for i in range(windowLength):
                pos = posStart + 1 + i
                if pos%stepSize == 0:
                    s += str(0.0) + "\n"
#                s += str(pos) + '  ' + str(0.0) + "\n"
            wigFile.write(s)
    
            wigFile.close()

            
                           
    avgCorrAll = cntCorrAll/cntTotAll #nrStepsAll
    print("Full genome, average prediction acc: %f" % avgCorrAll)
    
    if nrRepsAll > 0.5: #ie if there are any repeats recorded
        avgCorrRepAll = cntCorrRepAll/cntTotRepAll
        avgCorrNonRepAll = (cntCorrAll - cntCorrRepAll)/(cntTotAll -cntTotRepAll)
        print("Average prediction acc at repeats: %f and elsewhere: %f" % (avgCorrRepAll, avgCorrNonRepAll))
    else:
        print("No repeat sections were recorded in the genome data.")


    return avgPred




def predictAcrossGenome_II(frqModelFileName,
                        flankSizeFrqModel,
                        modelFileName, 
                        genomeData,
                        outputEncodedOneHot_b = 1,
                        outputEncodedInt_b = 0,
                        outputEncodedType = 'int8',
                        convertToPict_b = 0,
                        augmentWithRevComplementary_b = 0, 
                        customFlankSize_b = 0, 
                        customFlankSize = 50, 
                        shuffle_b = 0,
                        inner_b = 1, 
                        shuffleLength = 5,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 100,
                        save_predPlot_b = 0,
                        predPlotName = 'type_II',
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence (encoded, as
    read in by readGenome) by using the input estimated/trained model.
    Input:
    
    genomeData: tuple dna-sequence, repeat info as returned be readGenome
    '''
    
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"


    #load the frq-model's results:
    frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)          
            
    #load model trained to predict the truth value of the frq-model: 
    net = model_from_json(open(rootOutput + modelFileName).read())
    net.load_weights(rootOutput + modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
    sizeOutput = net.output_shape[1]

    genomeSeq, repeatInfoSeq =  genomeData

    lGenome = len(genomeSeq)
    
    print("Genome length: %d" % lGenome)

    #Read in the test data.
    #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
    Xt, Yt, Rt = getAllSamplesFromGenome(genomeData = genomeData, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)


#    root = r"/data/tkj375/DNA_proj/Inputs/"
#    root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
#    
#    tname= root + r"training.dat" #root + r"test.dat"
#    testDataInterval = [1000000, 2000000] 
#    testDataInterval = [0, 2000] 
#    Xt,Yt = getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
#  
#    print Xt.shape
#    print Yt.shape

#    Xt = Xt.astype('float32')
#    Yt = Yt.astype('float32')

    lSamples = len(Yt)
#
#    chunkSize = 10000
#    chunkNrSamples = chunkSize/(2*customFlankSize)
#    
#    nrOfPasses = lSamples/chunkSize
#
#    for i in range(nrOfPasses):
#        
#        for j in range(i*chunkSize, (i+1)*chunkSize):

    
    #Call the prediction
    pred = net.predict(Xt, batch_size = batchSize)
    
    print("Nr of sites: %d ; of which are predicted: %d" % (lSamples, pred.shape[0]))

    nrSteps = int((lSamples - windowLength)/stepSize)

    print("nrSteps ", nrSteps )
    
    cntTot = 0.0
    cntCorr = 0.0
    cntTotRep = 0.0
    cntCorrRep = 0.0
#    cntCorrA = 0.0 
#    cntCorrT = 0.0
#    cntCorrC = 0.0
#    cntCorrG = 0.0
    
#    cntA = 0
#    cntT = 0
#    cntC = 0
#    cntG = 0
    
    windowList = []
    windowArray = np.zeros(shape = windowLength, dtype = 'float32')
    avgPred = np.zeros(shape = nrSteps, dtype = 'float32')

    keyErrorCnt = 0
    for j in range(nrSteps):
        
        if j%1000 == 0:
            print("Now at step: ", j)
    
        #first window: read in the following windowLength worth of sites:
        if j == 0:
            
            for i in range(windowLength): #range(pred.shape[0]):
                
#                print " ".join(map(str,Yt[i])), " ".join(map(str,pred[i]))
            
                #prediction of whether frq-model right/wrong:
#                print "truth val predict:", pred[i]
                predIdx = np.argmax(pred[i])
                
#                print predIdx

                #frq-model's guess:
                charList = map(invOneHotLetter, Xt[i][(customFlankSize - flankSizeFrqModel):(customFlankSize + flankSizeFrqModel)])
#                print charList 
                word = ''
                word = word.join(charList)
#                print word
                try:
                    frqPred = frqModelDict[word]
#                    print frqPred
                except KeyError:
                    keyErrorCnt +=1 
#                    print "KeyError when reading from frqModelDict, key: ", word
                    frqPred = np.array([0.25, 0.25, 0.25, 0.25])
                    
                frqGuess = np.argmax(frqPred)
#                print "label: frqGuess: ", Yt[i], frqGuess
                
                if predIdx == 1: #frq model is wrong
                    
                    #make new guess: take the second most probable one acc to the frq model:
                    frqGuess = np.argmax(frqPred - delta4(np.argmax(frqPred)))
#                    print "new guess: ", frqGuess
                    
#                elif predIdx != 0:
#
#                    print "Warning: prediction of frq-model's truth value is not 0/1!"
                
#                else:
#                    
#                    print "Keep guess"
                
                if frqGuess == np.argmax(Yt[i]):
                    
                    cntCorr += 1.0
                    
                    if Rt[i] > 0.5:
                        cntCorrRep += 1.0
                    
                    windowList.append(1.0)
                    
                else:
                    
                    windowList.append(0.0)    
                    
                    
                cntTot += 1
            
            
            windowArray = np.asarray(windowList)
            avgPred[j] = np.mean(windowArray) #and j = 0
            
#            print len(windowList), windowList
            
        else:
            
            #remove first stepSize elt's from list
            for k in range(stepSize):
#                print k
                windowList.pop(0)
                
            #Append the windowLength - stepSize next elts:
            for l in range(stepSize): 

                i = j*stepSize + windowLength - stepSize + l                                 
                
                #prediction of whether frq-model right/wrong:
                predIdx = np.argmax(pred[i])
                
#                print predIdx

                charList = map(invOneHotLetter, Xt[i][(customFlankSize - flankSizeFrqModel):(customFlankSize + flankSizeFrqModel)])
#                print charList 
                word = ''
                word = word.join(charList)
#                print word
                try:
                    frqPred = frqModelDict[word]
#                    print frqPred
                except KeyError:
                    keyErrorCnt += 1
#                    print "KeyError when reading from frqModelDict, key: ", word
                    frqPred = np.array([0.25, 0.25, 0.25, 0.25])

                frqGuess = np.argmax(frqPred)
#                print "label: frqGuess: ", Yt[i], frqGuess
              
                if predIdx == 1: #frq model is wrong
                    
                    #make new guess: take the second most probable one acc to the frq model:
                    frqGuess = np.argmax(frqPred - delta4(np.argmax(frqPred)))
#                    print "new guess: ", frqGuess
                    
#                elif predIdx != 1:
#
#                    print "Warning: prediction of frq-model's truth value is not 0/1!"
#
#                else:
#                    
#                    print "Keep guess"                
                
                if frqGuess == np.argmax(Yt[i]):

                    cntCorr += 1
                    if Rt[i] > 0.5:
                        cntCorrRep += 1.0
                        cntTotRep  += 1.0
                        
                    windowList.append(1.0)
                        
                else:
                    
                    windowList.append(0.0)
                    
                    if Rt[i] > 0.5:
                        cntTotRep  += 1.0
                
                cntTot += 1

            windowArray = np.asarray(windowList)
            avgPred[j] = np.mean(windowArray)
                
    print("keyErrorCnt: ", keyErrorCnt)
        
    
    avgCorr = cntCorr/cntTot
    print("Average prediction acc : %f" % avgCorr)

    nrReps = np.sum(Rt)
    if nrReps > 0.5: #ie if there are any repeats recorded
        avgCorrRep = cntCorrRep/cntTotRep
        avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
        print("Average prediction acc at repeats: %f and elsewhere: %f" % (avgCorrRep, avgCorrNonRep))
    else:
        print("No repeat sections were recorded in the genome data.")
   
    if save_predPlot_b == 1:  
        
        plt.figure()       
        plt.plot(avgPred) 
        plt.savefig(rootOutput + modelFileName + '_predPlot_' + predPlotName + '.pdf' )    

    
    return avgPred
    

def predictAcrossGenomeDict_II(frqModelFileName,
                               flankSizeFrqModel,
                               modelFileName, 
                        genomeDictTuple,
                        outputEncodedOneHot_b = 1,
                        outputEncodedInt_b = 0,
                        outputEncodedType = 'int8',
                        convertToPict_b = 0,
                        augmentWithRevComplementary_b = 0, 
                        customFlankSize_b = 0, 
                        customFlankSize = 50, 
                        shuffle_b = 0,
                        inner_b = 1, 
                        shuffleLength = 5,
                        batchSize = 128,
                        windowLength = 100,
                        stepSize = 50, 
                        outputWig_b = 1,
                        wigFileName = '',
                        on_binf_b = 1):
    '''Predicts bases at each position across the input genome sequence (encoded, as
    read in by readGenome) by using the input estimated/trained model.
    Input: 
    genomeDictTuple: structure genome seq dict, repeat info dict as output by readGenome
    (genome DNA-sequence/repeat info structured in dictionary mapping each chromosome to 
    its sequence/repeat info).'''
    
    if on_binf_b == 1:
        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
    else:
        root = r"D:/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
        rootDevelopment = r"D:/Bioinformatics/various_python/theano/DNA_proj/development/"
        rootOutput = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_nets/"



    #load the frq-model's results:
    frqModelDict = getResultsFrqModel(fileName = frqModelFileName, flankSize = flankSizeFrqModel)          
            

    #load model trained to predict the truth value of the frq-model: 
    net = model_from_json(open(modelFileName).read())
    net.load_weights(modelFileName +'.h5')
    
    numI, letterShape = net.input_shape[-2:]
    sizeOutput = net.output_shape[1]

    cntCorrAll = 0.0
    cntTotAll = 0.0
    nrStepsAll = 0
    cntCorrRepAll = 0.0
    cntTotRepAll = 0.0
    nrRepsAll = 0
    
    chromoCnt = 0
    
    genomeDict, repeatInfoDict = genomeDictTuple
    
    for chromo in genomeDict.keys():
        
        chromoCnt += 1

        genomeData = genomeDict[chromo], repeatInfoDict[chromo]

        lGenome = len(genomeData[0])
        
        print("Chromo: %s genome length: %d" % (chromo,lGenome))
    
        #Read in the test data.
        #This includes a possible shuffling of the inner/outer flanks (or just the flanks) if desired:
        Xt, Yt, Rt = getAllSamplesFromGenome(genomeData = genomeData, outputEncodedType = outputEncodedType, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
    
    
    #    root = r"/data/tkj375/DNA_proj/Inputs/"
    #    root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
    #    
    #    tname= root + r"training.dat" #root + r"test.dat"
    #    testDataInterval = [1000000, 2000000] 
    #    testDataInterval = [0, 2000] 
    #    Xt,Yt = getData2(tname, letterShape, sizeOutput, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
    #  
        print("Shape flanks data across this chromo: ", Xt.shape)
        print("Shape midpoints data across this chromo: ", Yt.shape)
    
    #    Xt = Xt.astype('float32')
    #    Yt = Yt.astype('float32')
    
        lSamples = len(Yt)
    #
    #    chunkSize = 10000
    #    chunkNrSamples = chunkSize/(2*customFlankSize)
    #    
    #    nrOfPasses = lSamples/chunkSize
    #
    #    for i in range(nrOfPasses):
    #        
    #        for j in range(i*chunkSize, (i+1)*chunkSize):
    
        #Call the prediction
        pred = net.predict(Xt, batch_size = batchSize)
        
        print("Nr of sites: %d ; of which are predicted: %d" % (lSamples, pred.shape[0]))
    
        nrSteps = int((lSamples - windowLength)/stepSize)
    
        print("nrSteps ", nrSteps )
        
    #    print "nrSteps ", nrSteps 
        
        cntCorr = 0.0
        cntTot = 0.0
        cntCorrRep = 0.0
        cntTotRep = 0.0
#        cntCorrA = 0.0
#        cntCorrT = 0.0
#        cntCorrC = 0.0
#        cntCorrG = 0.0
#        
#        cntA = 0
#        cntT = 0
#        cntC = 0 
#        cntG = 0
        
        windowList = []
        windowArray = np.zeros(shape = windowLength, dtype = 'float32')
        avgPred = np.zeros(shape = nrSteps+1, dtype = 'float32')
    
        for j in range(nrSteps):
        
            #first window: read in the following windoLength worth of sites:
            if j == 0:
                
                for i in range(windowLength): #range(pred.shape[0]):
                    
    #                print " ".join(map(str,Yt[i])), " ".join(map(str,pred[i]))
                
                    predIdx = np.argmax(pred[i])
                    
    #                print predIdx
                    
                    if Yt[i][predIdx] > 0.5:
                        cntCorr += 1.0
                        if Rt[i] > 0.5:
                            cntCorrRep += 1.0
                        windowList.append(1.0)
                        
                    else:
                        windowList.append(0.0)
                        
                    cntTot += 1
                
                
                windowArray = np.asarray(windowList)
                avgPred[j] = np.mean(windowArray) #and j = 0
                
            else:
                
                #remove first stepSize elt's from list
                for k in range(stepSize):
                    windowList.pop(0)
                    
                #Append the windowLength - stepSize next elts:
                for l in range(windowLength - stepSize): 
                    i = j*stepSize + l
                    predIdx = np.argmax(pred[i])
                    if Yt[i][predIdx] > 0.5:
                            cntCorr += 1
                            if Rt[i] > 0.5:
                                cntCorrRep += 1.0
                                cntTotRep  += 1.0
                            windowList.append(1.0)
                            
                    else:
                        windowList.append(0.0)
                        if Rt[i] > 0.5:
                            cntTotRep  += 1.0
                    
                    cntTot += 1
    
                windowArray = np.asarray(windowList)
                avgPred[j] = np.mean(windowArray)
                
        #forcing the avgPred to be periodic (enabling the Fourier transform):
        avgPred[stepSize] = avgPred[0]
        print("Avg pred at 0: %f  and at nrSteps: %f" %(avgPred[0], avgPred[stepSize]) )
        
        plt.figure()
        plt.title(chromo + ' avg prediction')        
        plt.plot(avgPred) 
        plt.savefig(modelFileName + '_predPlot_' + chromo[30:70] + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )

        #Fourier transform it:
        fftAvgPred = fft(avgPred) #scipy fast Fourier transform
        print("Post fft: Avg pred at 0: %f  and at nrSteps: %f" %(avgPred[0], avgPred[stepSize]) )
        plt.figure()
        plt.title(chromo + ', fft avg prediction, excerpt')  
        start = int(nrSteps/34)
        end = int(nrSteps/33)
        plt.bar(range(start,end),fftAvgPred[start:end]) 
        plt.savefig(modelFileName + '_FourierTransformPredPlotZoom_2_' + chromo[30:70] + '_win' + str(windowLength) + '_step' + str(stepSize) + '.pdf' )
        
        
        avgCorr = cntCorr/cntTot
        print("Chromo: %s , average prediction acc : %f" % (chromo, avgCorr))

        nrReps = np.sum(Rt)
        if nrReps > 0.5: #ie if there are any repeats recorded
            avgCorrRep = cntCorrRep/cntTotRep
            avgCorrNonRep = (cntCorr - cntCorrRep)/(cntTot -cntTotRep)
            print("Chromo: %s, average prediction acc at repeats: %f and elsewhere: %f" % (chromo, avgCorrRep, avgCorrNonRep))
        else:
            prin( "Chromo: %s, no repeat sections were recorded in the genome data." % chromo)

        cntCorrAll += cntCorr
        cntTotAll += cntTot
        nrStepsAll += nrSteps
        cntCorrRepAll += cntCorrRep
        cntTotRepAll += cntTotRep
        nrRepsAll += nrReps
        
        #if desired write to wig-file (for custom track in UCSC genome browser);
        #there will be one file/track per chromosome 
        if outputWig_b == 1:
                
            if wigFileName == '':
                useWigFileName = modelFileName
            try:    
                useWigFileName = rootOutput + wigFileName  + '_pred_' + chromo + '_win' + str(windowLength) + '_step' + str(stepSize) + '.wig'
                #flush the wig-file by opening it and closing it again:
                wigFile = open(useWigFileName, 'w')
            except IOError: #file name may be prohibited; so try cutting down the chromo's name 
                useWigFileName = rootOutput + wigFileName  + '_pred_chromoNN' + str(chromoCnt) + '.wig'
                #flush the wig-file by opening it and closing it again:
                wigFile = open(useWigFileName, 'w')
            wigFile.close()
            wigFile = open(useWigFileName, 'a')
            print("For chromo: %s will write to wig-file: %s" % (chromo, useWigFileName )  )
        
        
            s = 'browser position ' + chromo + ':1-' + str(lSamples) + "\n"
            wigFile.write(s)
            s = 'browser hide all' + "\n"
            wigFile.write(s)
            #	200 base wide points graph at every 300 bases, 50 pixel high graph
            #	autoScale off and viewing range set to [0:1000]
            #	priority = 20 positions this as the second graph
            #	Note, one-relative coordinate system in use for this format
            s = 'track type=wiggle_0 name="fixedStep" description="fixedStep format" visibility=full autoScale=off viewLimits=0:1000 color=0,200,100 maxHeightPixels=100:50:20 graphType=points priority=20' + "\n"
            wigFile.write(s)
            s='fixedStep chrom=' + chromo + ' start=1' + ' step=' + str(stepSize) + "\n"
            wigFile.write(s)
            #loop across the range and write to file in chunks of e.g. 1000 lines
            #first and last windowLength worth of positions are left at 0 (there are no
            #predictions there):
            s = ''
            for i in range(windowLength):
                pos = i + 1
                if pos%stepSize == 0:
                    s += str(0.0) + "\n"
#                s += str(pos) + '  ' + str(0.0) + "\n"
            wigFile.write(s)
            s = ''
            posStart = pos
#            sumAvgPred = 0
            for j in range(nrSteps): #nrSteps
#                print avgPred[j]
                pos = posStart + 1 + j*stepSize
#                sumAvgPred += avgPred[j]
                s += str(round(avgPred[j], 6)) + "\n"
#                if j%100 == 0:
#                    avg = sumAvgPred/100
##                    s += str(round(avg, 6)) + "\n"
#                    sumAvgPred = 0
##                    s += str(pos) + '  ' + str(round(avg, 6)) + "\n" #for variableStep wig-type
                if j%100  == 0 and j > 0:
                    wigFile.write(s)
                    s = '' #reset
            #in case nrSteps is not divisible by 100, we need to write this last piece to file too:
            if j%100 != 0:
                wigFile.write(s)
            s = ''
            posStart = pos
            for i in range(windowLength):
                pos = posStart + 1 + i
                if pos%stepSize == 0:
                    s += str(0.0) + "\n"
#                s += str(pos) + '  ' + str(0.0) + "\n"
            wigFile.write(s)
    
            wigFile.close()

            
                           
    avgCorrAll = cntCorrAll/cntTotAll #nrStepsAll
    print("Full genome, average prediction acc: %f" % avgCorrAll)
    
    if nrRepsAll > 0.5: #ie if there are any repeats recorded
        avgCorrRepAll = cntCorrRepAll/cntTotRepAll
        avgCorrNonRepAll = (cntCorrAll - cntCorrRepAll)/(cntTotAll -cntTotRepAll)
        print("Average prediction acc at repeats: %f and elsewhere: %f" % (avgCorrRepAll, avgCorrNonRepAll))
    else:
        print("No repeat sections were recorded in the genome data.")


    return avgPred


###############################################################################
##        Models
###############################################################################

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
            testOnly_b = 0):
    
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
                    Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  exonicInfoBinaryFileName = exonicInfoBinaryFileName, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
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
            
    
    




def makeLSTMmodel_old(sequenceLength, letterShape = 4, outputSize = 4, batchSize = 100, hiddenUnits = [10] , return_sequences=True, stateful=True):
    '''
    network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    outputSize = the size of the output layer, here 4
    '''

    print('Build LSTM model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(hiddenUnits, return_sequences=return_sequences, stateful=stateful), batch_input_shape=(batchSize, sequenceLength, letterShape)))
    #model.add(LSTM(20, return_sequences=return_sequences, stateful=stateful, batch_input_shape=(batchSize, sequenceLength, letterShape)))
    model.add(LSTM(hiddenUnits, return_sequences=False, stateful=stateful))

#    model.add(LSTM(100,  stateful=True, batch_input_shape=(batchSize, sequenceLength, letterShape)))

#    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(sequenceLength, letterShape)))
#    model.add(LSTM(50, input_shape=(sequenceLength, letterShape), return_sequences=True))
#    model.add(LSTM(1000, input_shape=(sequenceLength, letterShape)))
    

#    model.add(TimeDistributed(Dense(letterShape, activation='sigmoid')))
#    model.add(Dense(outputSize, activation='softsign'))
    model.add(Dense(outputSize, activation='softsign'))
#    model.add(Activation('softmax'))

    print("... build model.")
        
    return model



def makeConv1DLSTMmodel_old(sequenceLength, letterShape, lengthWindows, nrFilters, sizeHidden, outputSize = 4, pool_b = 1, maxPooling_b = 0, poolAt = [2], dropoutLastLayer_b = 1, dropoutVal = 0.25):
    '''
    network model
    sequenceLength = lenght of the sequence (number of letters)
    letterShape = shape of letter encoding, here arrays of length 4
    lengthWindows = list of window lengths of the sliding windows in the cnn (order determines the layers)
    nrFilters =  list of the sizes of the outputs of each cnn layer, the "features" (ordered as the layers, ie corr to the lengthWindows list)
    sizeHidden = the size of the last flt layer
    outputSize = the size of the output layer, here 4
    '''

    print('Build Conv1d plus LSTM model...')    
    
    model = Sequential()
    
    #conv    = Conv1D(kernel_size=window, strides=1, filters=nrFilters[0], padding='valid', activation='relu')(inputs)
    print("ba ..")
    model.add(Conv1D(kernel_size=lengthWindows[0], strides=1, filters=nrFilters[0], padding='same', activation='relu', input_shape=(sequenceLength,letterShape)))
    print("..by")
    if pool_b == 1 and poolAt.count(0) > 0:
        if maxPooling_b == 1:
            model.add(MaxPooling1D())
        else:
            model.add(AveragePooling1D())
    
    for i in range(len(nrFilters)-1):    
    
        model.add(Conv1D(kernel_size=lengthWindows[i+1], strides=1, filters=nrFilters[i+1], padding='same', activation='relu')) #, input_shape=(sequenceLength,letterShape)))
        
        if pool_b == 1  and poolAt.count(i+1) > 0:
            if maxPooling_b == 1:
                model.add(MaxPooling1D())
            else:
                model.add(AveragePooling1D())
    
    
    model.add(Bidirectional(LSTM(sizeHidden, return_sequences=True))) #, input_shape=(sequenceLength, letterShape)))
    model.add(Flatten())
    model.add(Dropout(dropoutLastLayer_b*dropoutVal))
    model.add(Dense(outputSize, activation='softmax'))
    
    print("... build model.")
    
    return model




###########################################################

#if __name__ == '__main__':
#    if len(sys.argv) < 6: 
#        print "usage:",sys.argv[0]," trainingset testset  window,filter1,filter2,filter3,hdden sizeOutput model [epoch] "
#        sys.exit()
#    fname=sys.argv[1]
#    tname=sys.argv[2]
#    numHL=[int(x) for x in sys.argv[3].split(",") ]
#    window,nrHiddenUnits = numHL[0], numHL[-1]
#    nrFilters = numHL[1:-1]
#    sizeOutput=int(sys.argv[4])
#    modname=sys.argv[5]
#    try:
#       nrEpochs=int(sys.argv[6])
#    except:
#       nrEpochs=100
#  
#    letterShape = 4 # size of the word
#  
#    X,Y = getData2(fname,letterShape,sizeOutput,outputType=float)  
#    sizeInput = X.shape[1] 
#    
#    
#    Xt,Yt = getData2(tname,letterShape, sizeOutput,outputType=float)  
#    batch_size=min(64,max(1,len(X)/20))
#    #print batch_size, nrEpochs
#  
#    net = makeModel(sizeInput, letterShape, window, nrFilters, nrHiddenUnits, sizeOutput)
#    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    #sgd = SGD(lr=0.01)
#    net.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
#
#    #net.compile(loss='categorical_crossentropy', optimizer='Adam')
#    #net.compile(loss='binary_crossentropy', optimizer=sgd)
#    #xor.compile(loss="hinge", optimizer=sgd)
#    #xor.compile(loss="binary_crossentropy", optimizer=sgd)
#  
#
#    #net.fit(X, Y, batch_size=batch_size, nb_epoch=nrEpochs, show_accuracy=True, verbose=0)
#    history = net.fit(X, Y, batch_size=batch_size, nrEpochs=nrEpochs, verbose=0, validation_data=(Xt,Yt) )
#
#    json_string = net.to_json()
#    open(modname, 'w').write(json_string)
#    net.save_weights(modname+'.h5',overwrite=True)


#####################################################################################
## LeNet5 code: bulding conv2d model "by hand"
#####################################################################################

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 08:57:34 2018

@author: Copied from http://deeplearning.net/tutorial; code is on: http://deeplearning.net/tutorial/code/convolutional_mlp.py

This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

#import os
##import sys
#import timeit
#
##import numpy
#
#import theano
#import theano.tensor as T
#from theano.tensor.signal import pool
#from theano.tensor.nnet import conv2d
#
#from logistic_sgd_fromDLnet import LogisticRegression, load_data #trivial modidfication/CG
##from logistic_sgd import LogisticRegression, load_data
#from MLP_fromDLnet import HiddenLayer #trivial modidfication/CG
##from mlp import HiddenLayer
#
#
#class LeNetConvPoolLayer(object):
#    """Pool Layer of a convolutional network """
#
#    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
#        """
#        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
#
#        :type rng: np.random.RandomState
#        :param rng: a random number generator used to initialize weights
#
#        :type input: theano.tensor.dtensor4
#        :param input: symbolic image tensor, of shape image_shape
#
#        :type filter_shape: tuple or list of length 4
#        :param filter_shape: (number of filters, num input feature maps,
#                              filter height, filter width)
#
#        :type image_shape: tuple or list of length 4
#        :param image_shape: (batch size, num input feature maps,
#                             image height, image width)
#
#        :type poolsize: tuple or list of length 2
#        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
#        """
#
#        print image_shape[3]
#        print filter_shape[3]
#
#        assert image_shape[3] == filter_shape[3]
#        self.input = input
#
#        # there are "num input feature maps * filter height * filter width"
#        # inputs to each hidden unit
#        fan_in = np.prod(filter_shape[1:])
#        # each unit in the lower layer receives a gradient from:
#        # "num output feature maps * filter height * filter width" /
#        #   pooling size
#        fan_out = (np.prod(filter_shape[:3]) //
#                   np.prod(poolsize))
#        # initialize weights with random weights
#        W_bound = np.sqrt(6. / (fan_in + fan_out))
#        self.W = theano.shared(
#            np.asarray(
#                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
#                dtype=theano.config.floatX
#            ),
#            borrow=True
#        )
#
#        # the bias is a 1D tensor -- one bias per output feature map
#        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
#        self.b = theano.shared(value=b_values, borrow=True)
#
#        # convolve input feature maps with filters
#        conv_out = conv2d(
#            input=input,
#            filters=self.W,
#            filter_shape=filter_shape,
#            input_shape=image_shape
#        )
#
#        # pool each feature map individually, using maxpooling
#        pooled_out = pool.pool_2d(
#            input=conv_out,
#            ds=poolsize,
#            ignore_border=True
#        )
#
#        # add the bias term. Since the bias is a vector (1D array), we first
#        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
#        # thus be broadcasted across mini-batches and feature map
#        # width & height
#        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#
#        # store parameters of this layer
#        self.params = [self.W, self.b]
#
#        # keep track of model input
#        self.input = input
#
#
#
#
#
#def allInOne_Conv2DModel(loss = "categorical_crossentropy",
#            learningRate = 0.1,
#            nrTrainSamples = 100000,
#            trainDataInterval = [0,200000] , 
#            nrValSamples = 20000,
#            valDataInterval = [200000,400000],   
#            nrTestSamples = 20000,
#            testDataInterval = [400000, 600000], 
#            customFlankSize_b = 0, 
#            customFlankSize = 50,
#            genSamples_b = 0, 
#            genomeFileName = '',
#            outputEncodedOneHot_b = 0, #keep this!
#            outputEncodedInt_b = 1, #keep this!
#            onlyOneRandomChromo_b = 0,
#            avoidChromo = [],
#            genSamplesFromRandomGenome_b = 0, 
#            randomGenomeSize = 4500000, 
#            randomGenomeFileName = 'rndGenome.txt',
#            augmentWithRevComplementary_b = 0, 
#            batchSize = 128, 
#            nrEpochs = 100,
#            sizeOutput=4,
#            letterShape = 4, # size of the word
#            lengthWindows = [3, 3],
#            nrHiddenUnits = 50,
#            nrFilters = [50, 50],     
#            pool_b = 0,
#            maxPooling_b = 0,
#            poolAt = [],
#            poolStrides = 1,
#            shuffle_b = 0, 
#            inner_b = 1, 
#            shuffleLength = 5,
#            save_model_b = 1, 
#            modelName = 'ownSamples/EColi/modelConv2D_1', 
#            modelDescription = 'Conv 2D type ... to be filled in!',
#            on_binf_b = 1):
#                
#    if on_binf_b == 1:
#        root = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/Inputs/"
#        rootDevelopment = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/"
#        rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/"
#    else:
#        root = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/Inputs/"
#        rootDevelopment = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/development/"
#        rootOutput = r"C:/Users/Christian/Bioinformatics/various_python/theano/DNA_proj/results_nets/"
#
#
#    convertToPict_b = 0
#
#    
#    if genSamples_b > 0.5: #generate a set of random data acc to the sizes set
#
#        #if a genomeFileName is specified, use that genome:
#        if len(genomeFileName) > 0:
#            fromGenome_b = 1
#        else:
#            fromGenome_b = 0
#    
#        X,Y, genomeSeqSourceTrain = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = trainDataInterval[1] - trainDataInterval[0], flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
#        sizeInput = X.shape[1]
##        print "X shape", X.shape
#
#        avoidChromo.append(genomeSeqSourceTrain) #to avoid getting val data from the same chromo as the training data 
#        Xv,Yv, genomeSeqSourceVal = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = valDataInterval[1] - valDataInterval[0], flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
#        
#        avoidChromo.append(genomeSeqSourceVal) ##to avoid getting test data from the same chromo as the training and validation data 
#        Xt,Yt, genomeSeqSourceTest = genSamples_I(fromGenome_b = fromGenome_b, genomeFileName = genomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, onlyOneRandomChromo_b = onlyOneRandomChromo_b , avoidChromo = avoidChromo, nrSamples = testDataInterval[1] - testDataInterval[0], flankSize = customFlankSize, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
#        
#
##    elif genSamplesFromRandomGenome_b > 0.5: #generate a set of random data acc to the sizes set
##
##        #generate random genome of set size:   
##        genRandomGenome(length = randomGenomeSize, fileName = root + randomGenomeFileName, on_binf_b = on_binf_b) #will write the generated genome sequence to the file  
##
##        X,Y = genSamples_I(nrSamples = trainDataInterval[1] - trainDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
##        sizeInput = X.shape[1]
##
##        Xv,Yv = genSamples_I(nrSamples = valDataInterval[1] - valDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
##
##        Xt,Yt = genSamples_I(nrSamples = testDataInterval[1] - testDataInterval[0], fromGenome_b = 1, genomeFileName = randomGenomeFileName,  outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, convertToPict_b = convertToPict_b, flankSize = customFlankSize, augmentWithRevComplementary_b = augmentWithRevComplementary_b, getFrq_b = 0)
##        
##
##    else: #fetch the data from an appropriate source
##
##        #Using the getData2-fct to fetch data:  
##        fname=root + r"training.dat"
##        vname = root + r"validation.dat"
##        tname=root + r"test.dat"
##    
##        
##        X,Y = getData2(fname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = trainDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
##        sizeInput = X.shape[1]
##                
##        Xv,Yv = getData2(vname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = valDataInterval , outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)      
##            
##        Xt,Yt = getData2(tname, letterShape, sizeOutput, convertToPict_b = convertToPict_b, loadRecsInterval = testDataInterval, outputType=float, augmentWithRevComplementary_b = augmentWithRevComplementary_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, customFlankSize_b = customFlankSize_b, customFlankSize = customFlankSize)  
##    
#    
#    batch_size = min(batchSize,max(1,len(X)/20))
#    print batch_size, nrEpochs
#
#    rng = np.random.RandomState(23455)
#
#    train_set = X,Y
#    valid_set = Xv,Yv
#    test_set = Xt,Yt
#
##    train_set = intSeqConvertToOneHotSeq(X, inverse_b = 1),Y
##    valid_set = intSeqConvertToOneHotSeq(Xv, inverse_b = 1),Yv
##    test_set = intSeqConvertToOneHotSeq(Xt, inverse_b = 1),Yt
#
#
#    
#    # train_set, valid_set, test_set format: tuple(input, target)
#    # input is a numpy.ndarray of 2 dimensions (a matrix)
#    # where each row corresponds to an example. target is a
#    # numpy.ndarray of 1 dimension (vector) that has the same length as
#    # the number of rows in the input. It should give the target
#    # to the example with the same index in the input.
#
#    def shared_dataset(data_xy, borrow=True):
#        """ Function that loads the dataset into shared variables
#    
#        The reason we store our dataset in shared variables is to allow
#        Theano to copy it into the GPU memory (when code is run on GPU).
#        Since copying data into the GPU is slow, copying a minibatch everytime
#        is needed (the default behaviour if the data is not in a shared
#        variable) would lead to a large decrease in performance.
#        """
#        data_x, data_y = data_xy
##        print(data_x.shape, data_y.shape)
#        shared_x = theano.shared(np.asarray(data_x,
#                                               dtype=theano.config.floatX),
#                                 borrow=borrow)
##        print(shared_x.shape)
#        shared_y = theano.shared(np.asarray(data_y,
#                                               dtype=theano.config.floatX),
#                                 borrow=borrow)
##        print(shared_y.shape)
#        # When storing data on the GPU it has to be stored as floats
#        # therefore we will store the labels as ``floatX`` as well
#        # (``shared_y`` does exactly that). But during our computations
#        # we need them as ints (we use labels as index, and if they are
#        # floats it doesn't make sense) therefore instead of returning
#        # ``shared_y`` we will have to cast it to int. This little hack
#        # lets ous get around this issue
#        return shared_x, T.cast(shared_y, 'int32')
#
#
#    test_set_x, test_set_y = shared_dataset(test_set)
#    print("test set shapes:  ",   test_set_x.get_value(borrow=True).shape[0],  test_set_x.get_value(borrow=True).shape[1])
#    valid_set_x, valid_set_y = shared_dataset(valid_set)
#    train_set_x, train_set_y = shared_dataset(train_set)
##    return test_set_x
#    
#
#    # compute number of minibatches for training, validation and testing
#    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
#    n_train_batches //= batch_size
#    n_valid_batches //= batch_size
#    n_test_batches //= batch_size
#
#    # allocate symbolic variables for the data
#    index = T.lscalar()  # index to a [mini]batch
#
#    # start-snippet-1
#    x = T.matrix('x')   # the data is presented as rasterized images
#    y = T.ivector('y')  # the labels are presented as 1D vector of
#                        # [int] labels
#
#    ######################
#    # BUILD ACTUAL MODEL #
#    ######################
#    print('... building the model')
#
#    # Reshape matrix of rasterized images of shape (batch_size, flankSize * flankSize, 4*4)
#    # to a 4D tensor, compatible with our LeNetConvPoolLayer
#    # 16 (4*4) is the size of the letter-pair encodings.
#    layer0_input = x.reshape((batch_size, customFlankSize, customFlankSize, 16))
#
#    # Construct the first convolutional pooling layer:
#    # filtering reduces the image size to (50-3+1 , 50-3+1) = (48, 28)
#    # maxpooling reduces this further to (48/2, 48/2) = (24, 24)
#    # 4D output tensor is thus of shape (batch_size, nrFilters[0], 24, 24)
#    layer0 = LeNetConvPoolLayer(
#        rng,
#        input=layer0_input,
#        image_shape=(batch_size, customFlankSize, customFlankSize, 16),
#        filter_shape=(nrFilters[0], lengthWindows[0], lengthWindows[0], 16),
#        poolsize=(2, 2)
#    )
#    
##    print layer0.output.get_value(borrow=True).shape[0]
#
#    # Construct the second convolutional pooling layer
#    # filtering reduces the image size to (24-3+1, 24-3+1) = (22, 22)
#    # maxpooling reduces this further to (22/2, 22/2) = (11, 11)
#    # 4D output tensor is thus of shape (batch_size, nrFilters[1], 11, 11)
#    layer1 = LeNetConvPoolLayer(
#        rng,
#        input=layer0.output,
#        image_shape=(batch_size, 24, 24, 16),
#        filter_shape=(nrFilters[1], lengthWindows[1], lengthWindows[1], 16),
#        poolsize=(2, 2)
#    )
#    
##    print layer1.output.shape
#
#    # the HiddenLayer being fully-connected, it operates on 2D matrices of
#    # shape (batch_size, num_pixels, 16) (i.e matrix of rasterized images).
#    # This will generate a matrix of shape (batch_size, nrFilters[1] * 11 * 11 * 16)
#    layer2_input = layer1.output.flatten(2)
#
#    # construct a fully-connected sigmoidal layer
#    layer2 = HiddenLayer(
#        rng,
#        input=layer2_input,
#        n_in=nrFilters[1] * 4 * 4,
#        n_out=50,
#        activation=T.tanh
#    )
#
#    # classify the values of the fully-connected sigmoidal layer
#    layer3 = LogisticRegression(input=layer2.output, n_in=50, n_out=4)
#
#    # the cost we minimize during training is the NLL of the model
#    cost = layer3.negative_log_likelihood(y)
#
#    # create a function to compute the mistakes that are made by the model
##    test_model = theano.function(
##        [index],
##        layer3.errors(y),
##        givens={
##            x: seqToPict(inputArray = test_set_x[index * batch_size: (index + 1) * batch_size]),
##            y: test_set_y[index * batch_size: (index + 1) * batch_size]
##        }
##    )
#
#
#    test_model = theano.function(
#        [index],
#        layer3.errors(y),
##        givens={x: map(seqToPict, map(intSeqConvertToOneHotSeq(test_set_x[index * batch_size: (index + 1) * batch_size]))), 
#        givens={x: test_set_x[index * batch_size: (index + 1) * batch_size], 
#                y: test_set_y[index * batch_size: (index + 1) * batch_size]}
#
#    )
#
#    validate_model = theano.function(
#        [index],
#        layer3.errors(y),
#        givens={
#            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
#            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
#        }
#    )
#
#    # create a list of all model parameters to be fit by gradient descent
#    params = layer3.params + layer2.params + layer1.params + layer0.params
#
#    # create a list of gradients for all model parameters
#    grads = T.grad(cost, params)
#
#    # train_model is a function that updates the model parameters by
#    # SGD Since this model has many parameters, it would be tedious to
#    # manually create an update rule for each model parameter. We thus
#    # create the updates list by automatically looping over all
#    # (params[i], grads[i]) pairs.
#    updates = [
#        (param_i, param_i - learningRate * grad_i)
#        for param_i, grad_i in zip(params, grads)
#    ]
#
#    train_model = theano.function(
#        [index],
#        cost,
#        updates=updates,
#        givens={
#            x: map(seqToPict, train_set_x[index * batch_size: (index + 1) * batch_size]),
#            y: train_set_y[index * batch_size: (index + 1) * batch_size]
#        }
#    )
#    # end-snippet-1
#
#    ###############
#    # TRAIN MODEL #
#    ###############
#    print('... training')
#    # early-stopping parameters
#    patience = 10000  # look as this many examples regardless
#    patience_increase = 2  # wait this much longer when a new best is
#                           # found
#    improvement_threshold = 0.995  # a relative improvement of this much is
#                                   # considered significant
#    validation_frequency = min(n_train_batches, patience // 2)
#                                  # go through this many
#                                  # minibatche before checking the network
#                                  # on the validation set; in this case we
#                                  # check every epoch
#
#    best_validation_loss = np.inf
#    best_iter = 0
#    test_score = 0.
#    start_time = timeit.default_timer()
#
#    epoch = 0
#    done_looping = False
#
#    while (epoch < nrEpochs) and (not done_looping):
#        epoch = epoch + 1
#        for minibatch_index in range(n_train_batches):
#
#            iter = (epoch - 1) * n_train_batches + minibatch_index
#
#            if iter % 100 == 0:
#                print('training @ iter = ', iter)
#            cost_ij = train_model(minibatch_index)
#
#            if (iter + 1) % validation_frequency == 0:
#
#                # compute zero-one loss on validation set
#                validation_losses = [validate_model(i) for i
#                                     in range(n_valid_batches)]
#                this_validation_loss = np.mean(validation_losses)
#                print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                      (epoch, minibatch_index + 1, n_train_batches,
#                       this_validation_loss * 100.))
#
#                # if we got the best validation score until now
#                if this_validation_loss < best_validation_loss:
#
#                    #improve patience if loss improvement is good enough
#                    if this_validation_loss < best_validation_loss *  \
#                       improvement_threshold:
#                        patience = max(patience, iter * patience_increase)
#
#                    # save best validation score and iteration number
#                    best_validation_loss = this_validation_loss
#                    best_iter = iter
#
#                    # test it on the test set
#                    test_losses = [
#                        test_model(i)
#                        for i in range(n_test_batches)
#                    ]
#                    test_score = np.mean(test_losses)
#                    print(('     epoch %i, minibatch %i/%i, test error of '
#                           'best model %f %%') %
#                          (epoch, minibatch_index + 1, n_train_batches,
#                           test_score * 100.))
#
#            if patience <= iter:
#                done_looping = True
#                break
#
#    end_time = timeit.default_timer()
#    print('Optimization complete.')
#    print('Best validation score of %f %% obtained at iteration %i, '
#          'with test performance %f %%' %
#          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
##    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
#
#
##    # list all data in history
##    print(history.history.keys())
##    # summarize history for accuracy
##    plt.figure()
##    plt.plot(history.history['acc'])
##    plt.plot(history.history['val_acc'])
##    plt.title('model accuracy')
##    plt.ylabel('accuracy')
##    plt.xlabel('epoch')
##    plt.legend(['train', 'test'], loc='upper left')
##    plt.savefig(rootOutput + modelName + '_training_validation_acc_vs_epoch.pdf')
###    plt.show()
##    # summarize history for loss
##    plt.figure()
##    plt.plot(history.history['loss'])
##    plt.plot(history.history['val_loss'])
##    plt.title('model loss')
##    plt.ylabel('loss')
##    plt.xlabel('epoch')
##    plt.legend(['train', 'test'], loc='upper left')
##    plt.savefig(rootOutput + modelName + '_training_validation_loss_vs_epoch.pdf')
###    plt.show()
##
##    #test it:
##    score, acc = net.evaluate(Xt,Yt, batch_size=batch_size, verbose=1)
##    print('Test score:', score)
##    print('Test accuracy:', acc)
#    
##    if save_model_b == 1:
##         
##        json_string = net.to_json()
##        open(rootOutput + modelName, 'w').write(json_string)
##        net.save_weights(rootOutput + modelName + '.h5',overwrite=True)
#        
#        
#        
#    #Write run-data to txt-file for documentation of the run:
#    runDataFileName = rootOutput + modelName + '_runData.txt'
#    runDataFile = open(runDataFileName, 'w') #Obs: this will overwrite an existing file with the same name
#    
#    s = "Parameters used in this run of the Python code for the deepDNA-project." + "\n"   
#    s += modelDescription  + "\n"   
#    if save_model_b == 1:
#        s+= 'Model data obtained after training the model are recorded in: ' +  rootOutput + modelName + ' and ' + rootOutput + modelName + '.h5\n' 
#    runDataFile.write(s)
#    
#    s = '' #reset
#    runDataFile.write(s + "\n") #insert blank line
#    #Which genome data were used?:
#    if genSamples_b > 0.5:
#        s = "Samples generated with python code from real genome." + "\n"
#        s += "Genome data in file: " + genomeFileName + "\n"
#        s += "Letters are one-hot encoded" + "\n"
#        s += "nrTrainSamples:", str(nrTrainSamples)  + "\n"
#        s += "trainDataInterval:", str(trainDataInterval)  + "\n" 
#        s += "nrValSamples:", str(nrValSamples)  + "\n"
#        s += "valDataInterval:", str(valDataInterval)  + "\n"   
#        s += "nrTestSamples:", str(nrTestSamples)  + "\n"
#        s += "testDataInterval:", str(testDataInterval)  + "\n" 
#        if onlyOneRandomChromo_b == 1:
#            s += "Only read in data from one randomly chosen chromosome per task:"  + "\n"
#            s += "Train data from chromosome: " + genomeSeqSourceTrain  + "\n"
#            s += "Validation data from chromosome: " + genomeSeqSourceVal  + "\n"
#            s += "Test data from chromosome: " + genomeSeqSourceTest  + "\n"
#            s += "Avoided data from these chromosomes: " +  str(avoidChromo)  + "\n"
#        else:
#            s += "Read in the whole genome sequence" + "\n"
#        s += "shuffle_b = " + str(shuffle_b) + "\n"
#        s += "inner_b = " + str(inner_b) + "\n"
#        s += "shuffleLength = " + str(shuffleLength) +  "\n"
#        s += "nrTrainSamples:", str(nrTrainSamples)  + "\n"
#        s += "trainDataInterval:", str(trainDataInterval)  + "\n" 
#        s += "nrValSamples:", str(nrValSamples)  + "\n"
#        s += "valDataInterval:", str(valDataInterval)  + "\n"   
#        s += "nrTestSamples:", str(nrTestSamples)  + "\n"
#        s += "testDataInterval:", str(testDataInterval)  + "\n" 
#     
#    elif genSamplesFromRandomGenome_b > 0.5:
#        
#        s = "Samples from random genome, all generated with python code." + "\n"
#        s += "Genome data in file: " + randomGenomeFileName
#    
#    else: #fetch the data from an appropriate source
#
#        s = "Pre-generated samples (ie not generated with the python code.)" + "\n"
#        s += "Training samples from: " + fname  + "\n"
#        s += "Validation samples from: " + vname  + "\n"
#        s += "Test samples from: " + tname  + "\n"
#        s += "shuffle_b = " + str(shuffle_b) + "\n"
#        s += "inner_b = " + str(inner_b) + "\n"
#        s += "shuffleLength = " + str(shuffleLength) +  "\n"
#        
#    runDataFile.write(s)
#    
#    s = '' #reset
#    runDataFile.write(s + "\n") #insert blank line
#    #various params:    
#    s= 'loss = "categorical_crossentropy"\n' 
#    s += 'trainDataInterval: ' + str(trainDataInterval) + "\n"
#    s += 'valDataInterval: ' + str(valDataInterval) + "\n"
#    s += 'testDataInterval: ' + str(testDataInterval) + "\n" 
#    s += 'customFlankSize_b: ' + str(customFlankSize_b) + "\n" 
#    s += 'customFlankSize: ' + str(customFlankSize) + "\n" 
#    s += 'genSamples_b: ' + str(genSamples_b) + "\n" 
#    s += 'genomeFileName: ' + genomeFileName + "\n" 
#    s += "outputEncodedOneHot_b: " + str(outputEncodedOneHot_b) + "\n" 
#    s += "outputEncodedInt_b: " + str(outputEncodedInt_b) + "\n" 
#    s += "onlyOneRandomChromo_b: " + str(onlyOneRandomChromo_b)  + "\n" 
#    s += "avoidChromo: " + str(avoidChromo)  + "\n" 
#    s += 'genSamplesFromRandomGenome_b: ' + str(genSamplesFromRandomGenome_b) + "\n" 
#    s += 'randomGenomeSize: ' + str(randomGenomeSize) + "\n" 
#    s += 'randomGenomeFileName: ' + randomGenomeFileName + "\n" 
#    s += 'augmentWithRevComplementary_b: ' + str(augmentWithRevComplementary_b) + "\n" 
#    s += 'learning rate: ' + str(learningRate)  + "\n" 
#    s += 'batchSize: ' + str(batchSize) + "\n"
#    s += 'pool_b: ' +  str(pool_b) + "\n"
#    s += 'maxPooling_b: ' +  str(maxPooling_b) + "\n"
#    s += 'poolAt: ' +  str(poolAt) + "\n"
#    s += 'nrEpochs: ' + str(nrEpochs) + "\n" 
#    s += 'sizeOutput: ' + str(sizeOutput) + "\n" 
#    s += 'letterShape: ' + str(letterShape) + "\n" 
#    s += 'save_model_b: ' + str(save_model_b) + "\n" 
#    s += 'modelName: ' + modelName + "\n" 
#    s += 'on_binf_b: ' + str(on_binf_b) + "\n" 
#    
#    runDataFile.write(s)
#        
#    #Params for net:
#    s = '' #reset
#    runDataFile.write(s + "\n") #insert blank line
#    s = 'lengthWindows: ' + str(lengthWindows)  + "\n" 
#    s += 'nrHiddenUnits: ' + str(nrHiddenUnits)  + "\n" 
#    s += 'nrFilters: ' + str(nrFilters)
#    
#    runDataFile.write(s)
#    
#    runDataFile.close()
#        
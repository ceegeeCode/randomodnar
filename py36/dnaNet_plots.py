# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 02:10:54 2019

@author: Christian Grønbæk


For creating various plots of NN models' performance.
"""

'''
Usage:

import dnaNet_plots as plots


##############
# Human
##############

rootOutput_hg38 =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"

bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM1/modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50',
rootOutput_hg38 + '/LSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00',
rootOutput_hg38 + '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [2,1,1]
lastRepeatNrList = [99, 176, 199] #107?, 207?
epochsPerRepeatList = [100. -1, 100-1, 100-1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM1', 'LSTM4', 'LSTM4P']
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList)


#Similar plots, but for one model at a time:
#LSTM1
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM1/modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50']

batchSizeRatioList = [2]
lastRepeatNrList = [99]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM1']
fileNameAcc = 'LSTM1_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM1_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#LSTM4
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [176]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM4']
fileNameAcc = 'LSTM4_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#LSTM4S
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM4S/trainTestSplit_80_20/modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [199]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM4S']
fileNameAcc = 'LSTM4S_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4S_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#mouseLSTM4
bigLoopIterNr = 0
modelFileNameList = [ r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/" + r'mouseLSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [193]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['mouseLSTM4']
fileNameAcc = 'mouseLSTM4_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'mouseLSTM4_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#LSTM4P
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [199]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM4P']
fileNameAcc = 'LSTM4P_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4P_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#In pairs:


#LSTM1 and LSTM4P
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM1/modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50',
rootOutput_hg38 + '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [2,1]
lastRepeatNrList = [99,  199] #107?, 207?
epochsPerRepeatList = [100. -1,  100-1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM1',  'LSTM4P']
fileNameAcc = 'LSTM1_LSTM4P_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM1_LSTM4P_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#LSTM4 and LSTM4P
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00',
rootOutput_hg38 + '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1,1]
lastRepeatNrList = [176, 199]
epochsPerRepeatList = [100. -1, 100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM4',  'LSTM4P']
fileNameAcc = 'LSTM4_LSTM4P_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4_LSTM4P_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#LSTM4S and LSTM4P
bigLoopIterNr = 0
modelFileNameList = [rootOutput_hg38 + '/LSTM4S/trainTestSplit_80_20/modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00',
rootOutput_hg38 + '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1,1]
lastRepeatNrList = [199, 199]
epochsPerRepeatList = [100. -1, 100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/'
modelNameList = ['LSTM4S',  'LSTM4P']
fileNameAcc = 'LSTM4S_LSTM4P_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4S_LSTM4P_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



#Plots/tables for monitoring the various estimates of the accuracy
rootOutputList = [r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
, r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
, r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/"] 


modelNameList = ['LSTM1', 'LSTM4', 'LSTM4P', 'LSTM4S', 'LSTM11', 'mouseLSTM4']

modelFileNameList = [r'/modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr'
, r'/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr'
, r'/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr'
, r'/trainTestSplit_80_20/modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr'
, r'/modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr'
, r'/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr']

modelNameChangeDict = {'LSTM1':'LSTM200', 'LSTM4':'LSTM50', 'LSTM4P':'LSTM50P', 'LSTM4S':'LSTM50S', 'LSTM11':'LSTM200early', 'mouseLSTM4':'mouseLSTM50'}

repeatNrList = [99, 176, 199, 184 , 15, 193]
accDictFileNameList = [r'/notAvgRevCompl/accuracyByAnnoDictionary'
, r'/trainTestSplit_80_20/notAvgRevCompl/accuracyByAnnoDictionary'
, r'/trainTestSplit_80_20/notAvgRevCompl/accuracyByAnnoDictionary'
, r'/trainTestSplit_80_20/notAvgRevCompl/accuracyByAnnoDictionaryInChromoPartition'
, r'/notAvgRevCompl/accuracyByAnnoDictionary'
, r'/trainTestSplit_80_20/notAvgRevCompl/accuracyByAnnoDictionary']
#nr of train samples -- first repeat is indexed 0, so must add 1:
#LSTM1: 2*(99+1)*5000000
#LSTM4: (176+1)*5000000
#LSTM4S: (184+1)*5000000
#LSTM11: 2*(15+1)*5000000
#mouseLSTM4: (193+1)*5000000

sampleSizeList = [10000000, 5000000, 5000000, 5000000, 10000000, 5000000]
sampleFactorList = [1., 0.8, 1.0, 0.8, 1., 0.8]

rootOutputPlot = r"/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/performancePlots/"
offsetValue = 0.0
plot_b = False
estimateByModelDict, samplingInfoByModelDict = plots.multimodelTrainTestFigures(rootOutputList = rootOutputList, modelNameList = modelNameList, modelFileNameList = modelFileNameList, modelNameChangeDict = modelNameChangeDict, repeatNrList = repeatNrList, accDictFileNameList= accDictFileNameList, sampleSizeList = sampleSizeList, sampleFactorList = sampleFactorList, rootOutputPlot = rootOutputPlot , offsetValue = offsetValue, plot_b = plot_b)

#make a tex table holding the results, accuracy/estimate:
rowNames = modelNameList
captionText = 'The accuracy obtained in training (acc last train, the average accuracy over the defining training round), validation (acc last val, the validation done after the definin training round), on the full genome (acc all) and our cautious estimate (acc est). '
fileName = 'table_accuracy_estimates.txt'
stats.makeTexTable(inputDict = estimateByModelDict , rowColHeading = 'model/level', rowNames = rowNames, captionText = captionText, rootOutput = rootOutputPlot, fileName = fileName ) 

#make a tex table holding the sampling info, nr repeats, nr samples etc:
rowNames = modelNameList
columnNames = ['nTotal', 'nTotalTrain', 'nrRepeats',  'nSamplesPerRepeat', 'nrSamples', 'expectedNrDiffSamples', 'expectedCoverage']
nameChangeDict = {'nTotal':'#all', 'nTotalTrain':'#all train', 'nrRepeats':'#rounds', 'nrSamples':'#samples', 'nSamplesPerRepeat':'#samples per round', 'expectedNrDiffSamples':'#diff samples', 'expectedCoverage':'coverage'}
captionText = 'Some sampling statistics for the training of the models.'
fileName = 'table_samplingInfoByModel.txt'
stats.makeTexTable(inputDict = samplingInfoByModelDict , rowColHeading = 'model', rowNames = rowNames, columnNames = columnNames,   nameChangeDict = nameChangeDict, captionText = captionText, rootOutput = rootOutputPlot, fileName = fileName ) 


#############
# Yeast
#############

rootOutput_R64 =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/on_R64"

#LSTM4
bigLoopIterNr = 0
modelFileNameList = [rootOutput_R64 + '/LSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [59]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/performancePlots/'
modelNameList = ['LSTM4']
fileNameAcc = 'LSTM4_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#LSTM4P and LSTM4R
bigLoopIterNr = 0
modelFileNameList = [rootOutput_R64+ '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00',
rootOutput_R64 + '/LSTM4R/trainTestSplit_70_30/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1,1]
lastRepeatNrList = [250,  170] #107?, 207?
epochsPerRepeatList = [10. -1,  10. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/performancePlots/'
modelNameList = ['LSTM4P',  'LSTM4R']
fileNameAcc = 'LSTMP_LSTM4R_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTMP_LSTM4R_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#LSTM4Q
bigLoopIterNr = 0
modelFileNameList = [rootOutput_R64 + '/LSTM4Q/trainTestSplit_70_30/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [300] #107?, 207?
epochsPerRepeatList = [10. -1,  10. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/performancePlots/'
modelNameList = ['LSTM4Q']
fileNameAcc = 'LSTMQ_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTMQ_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#LSTM4P,Q and R:
bigLoopIterNr = 0
modelFileNameList = [rootOutput_R64+ '/LSTM4P/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00',
rootOutput_R64 + '/LSTM4R/trainTestSplit_70_30/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00',
rootOutput_R64 + '/LSTM4Q/trainTestSplit_70_30/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']
]

batchSizeRatioList = [1,1,1]
lastRepeatNrList = [250,  250, 250] #107?, 207?
epochsPerRepeatList = [10. -1,  10. -1,  10. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/yeast/performancePlots/'
modelNameList = ['LSTM4P',  'LSTM4R', 'LSTM4Q']
fileNameAcc = 'LSTMP_LSTM4R_LSTM4Q_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTMP_LSTM4R_LSTM4Q_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )


#############
# Droso
#############

rootOutput_r6_18 =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/on_r6.18"

#LSTM4
bigLoopIterNr = 0
modelFileNameList = [rootOutput_r6_18 + '/LSTM4/trainTestSplit_80_20/modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00']

batchSizeRatioList = [1]
lastRepeatNrList = [168]
epochsPerRepeatList = [100. -1] #you have to look up nrEpochs per repeat (the log file for the run, runData ); subtract 1 since epoch 1 in each repeat is apperently not saved

rootOutput = r'/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/drosophila/performancePlots/'
modelNameList = ['LSTM4']
fileNameAcc = 'LSTM4_total_trainTest_acc_vs_allEpochs'
fileNameLoss = 'LSTM4_total_trainTest_loss_vs_allEpochs'
plots.collectivePerfPlot(modelFileNameList = modelFileNameList, batchSizeRatioList = batchSizeRatioList, lastRepeatNrList = lastRepeatNrList, epochsPerRepeatList = epochsPerRepeatList, rootOutput = rootOutput, modelNameList = modelNameList, fileNameAcc= fileNameAcc, fileNameLoss = fileNameLoss )



###########################################################################################################
#doodles:
###########################################################################################################

rootOutput_hg38 =  r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/"
modelFileNameList = [rootOutput_hg38 + 'modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50']


modelFileName = modelFileNameList[0]
loadFile = modelFileName + '_bigLoopIter' + str(bigLoopIterNr) + '_repeatNr' + str(106) + '_testing_loss_vs_epoch.p'
lossThisRepeat = pickle.load(open( loadFile, "rb"))

loadFile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/' + 'modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr99_testing_acc_vs_epoch.p'
testAccThisRepeat = pickle.load(open( loadFile, "rb"))

loadFile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/' + 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176_training_acc_vs_epoch.p'
trainAccThisRepeat = pickle.load(open( loadFile, "rb"))

loadFile = r'/binf-isilon/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4S/trainTestSplit_80_20/' + 'modelLSTM_2LayerConv2LayerLstm1LayerDense50_flanks50_win3_filters64and256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr184_training_acc_vs_epoch.p'
trainAccThisRepeat = pickle.load(open( loadFile, "rb"))


n = 1000000
N = 3000000
tolerance = 0.0001
nrOfDiffSamples = plots.expectedNrOfDiffSamples(n, N, tolerance)
nrOfDiffSamples


#glue ...
initRepeatNr = 61
fileName = rootOutput_hg19 + 'modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr' + str(initRepeatNr) + '_testing_acc_vs_epoch.p'
hist = pickle.load(open( fileName, "rb"))
print len(hist) 
print hist
lastRepeatNr = 242  
for i in range(initRepeatNr + 1, lastRepeatNr +1):
    newHist = []
    newHist = [hist[j] for j in range(len(hist))]
    #load the hist (from initRepeatNr + 1 to i)
    loadFile = rootOutput_hg19 + 'modelLSTM__1Conv2LayerLstm_flanks200_win4_stride1_overlap0_dropout00_bigLoopIter0_repeatNr' + str(i) + '_testing_acc_vs_epoch.p'
    histUpToHere = pickle.load(open( loadFile, "rb"))
    #print len(histUpToHere)
    
    #extend and re-save:
    newHist.extend(histUpToHere)
#    print newHist
    print i, len(newHist)
    
    #re-save
    pickle.dump(newHist, open( loadFile, "wb"))
    
###########################################################################################################
    
#Tex-string for LSTM4 vs mouseLSTM4 model-model plots: 
chromosomeList = [ 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr1']
chromosomeList = chromosomeList[::-1]
rootFileNamesList = ['LSTM4_vs_mouseLSTM4_refPredictions_hist2dPlot_','LSTM4_vs_mouseLSTM4_refPredictions_annotated_hist2dPlot_gene_','LSTM4_vs_mouseLSTM4_refPredictions_annotated_hist2dPlot_cds_','LSTM4_vs_mouseLSTM4_refPredictions_annotated_hist2dPlot_RmskAndTrf_']
caption = 'LSTM4 vs mouseLSTM4'    
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForLSTM4vsmouseLSTM4.txt'
texString = plots.texForMultiPlotByChromosome(chromosomeList = chromosomeList, rootFileNamesList = rootFileNamesList, caption = caption, rootOutput = rootOutput, fileNameOut = fileNameOut)    

#LSTM1 vs Markov
chromosomeList = [ 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr1']
chromosomeList = chromosomeList[::-1]
rootFileNamesList = ['Markov_k14_vs_LSTM1_refPredictions_hist2dPlot_','Markov_k14_vs_LSTM1_refPredictions_annotated_hist2dPlot_gene_','Markov_k14_vs_LSTM1_refPredictions_annotated_hist2dPlot_cds_','Markov_k14_vs_LSTM1_refPredictions_annotated_hist2dPlot_RmskAndTrf_']
caption = 'Markov k=14 vs LSTM1'    
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForMarkovVsLSTM1.txt'

texString = plots.texForMultiPlotByChromosome(chromosomeList = chromosomeList, rootFileNamesList = rootFileNamesList, caption = caption, rootOutput = rootOutput, fileNameOut = fileNameOut)    


###########################################################################################################

#Tex-string for multi Fourier plots:

chromosomeList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11','hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':40, 'hg38_chr21':40, 'hg38_chr20':63, 'hg38_chr19':57, 'hg38_chr18':80, 'hg38_chr17':82, 'hg38_chr16':90, 'hg38_chr15':83, 'hg38_chr14':90, 'hg38_chr13':97, 'hg38_chr12':132, 'hg38_chr11':134, 'hg38_chr10':132, 'hg38_chr9':137, 'hg38_chr8':144, 'hg38_chr7':158, 'hg38_chr6':170, 'hg38_chr5':180, 'hg38_chr4':190, 'hg38_chr3':197, 'hg38_chr2':241, 'hg38_chr1':247}
#chromosomeList = [ 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr1']
chromosomeList = chromosomeList[::-1]

#chromosomeList = [ 'hg38_chr20']

#generate sub-string list:
subListDict = plots.genSubListDict(chromosomeBaseDict = chromosomeDict, subListType = 0)
#gen text:
rootFileNamePart = ['Fourier_norm_w1000_step100_',  '_seg1000000_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForFourierHumanFouriersAllChromos.txt'
#caption = ['GC bias. Fouriers on GC/AT content arrays for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
caption = ['LSTM1. Fouriers on reference-base probability for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
plots.texForMultiPlot_ChromosomeBySub(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart, subListDict = subListDict, plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)


#mouse
#selected
chromosomeList = ['m38_chr11', 'm38_chr7', 'm38_chr5', 'm38_chr3']
chromosomeDict = {'m38_chr3':160, 'm38_chr5':150, 'm38_chr7':144, 'm38_chr11':121}

#all
chromosomeList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
chromosomeDict = { 'm38_chr1':194,'m38_chr2':181,'m38_chr3':159, 'm38_chr4':155, 'm38_chr5':150, 'm38_chr6':148,'m38_chr7':144,'m38_chr8':128,'m38_chr9':123,'m38_chr10':129,'m38_chr11':121,'m38_chr12':119,'m38_chr13':119,'m38_chr14':123,'m38_chr15':103,'m38_chr16':97,'m38_chr17':93,'m38_chr18':89,'m38_chr19':60}


#generate sub-string list:
subListDict = plots.genSubListDict(chromosomeBaseDict = chromosomeDict, subListType = 0)
#gen text:
rootFileNamePart = ['Fourier_norm_w1000_step100_',  '_seg1000000_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForMouseGCbiasFouriersAllChromos.txt'
caption = ['Mouse GC/AT content. Fouriers on reference-base probability for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
plots.texForMultiPlot_ChromosomeBySub(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart, subListDict = subListDict, plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)


#One plot of all the all-segments plots (ie all chromos)
#gen text:
rootFileNamePart = ['Fourier_norm_w5000_step100_', '_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForMouseLSTM4FouriersAllChromosTotOnePlot_hiFrqs.txt'
caption = 'Mouse LSTM4. Fouriers on reference-base probability. Each plot covers one chromosome, listed in increasing order (chr1 to chr19). The genome string is divided in adjacent segments of 1Mb (per chromosome); each plot shows the results for all segments in the chromosome (with ratio of qualified positions $>$ 0.9).'
plots.texForOnePlot_AllChromosomes(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart,  plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)



#yeast
chromosomeList = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
chromosomeDict = {'R64_chr1':1, 'R64_chr2':7, 'R64_chr3':3, 'R64_chr4':15, 'R64_chr5':4, 'R64_chr6':1, 'R64_chr7':9, 'R64_chr8':4,'R64_chr9':3, 'R64_chr10':6, 'R64_chr11':6, 'R64_chr12':9,'R64_chr13':9, 'R64_chr14':6, 'R64_chr15':9, 'R64_chr16':9}

#chromosomeList = ['R64_chr4']
#generate sub-sring list:
subListDict = plots.genSubListDict(chromosomeBaseDict = chromosomeDict, subListType = 0, step = 5)
#gen text:
rootFileNamePart = ['Fourier_norm_w200_step10_',  '_seg100000_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForYeastLSTM4FouriersAllChromos.txt'
caption = ['Yeast LSTM4. Fouriers on reference-base probability for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
plots.texForMultiPlot_ChromosomeBySub(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart, subListDict = subListDict, plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)


#One plot of all the all-segments plots (ie all chromos)
#gen text:
rootFileNamePart = ['Fourier_norm_w500_step10_', '_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForYeastLSTM4FouriersAllChromosTotOnePlot_hiFrqs.txt'
caption = 'Yeast LSTM4. Fouriers on reference-base probability, higher frequency range. Each plot covers one chromosome, listed in increasing order (chr1 to chr16). The genome string is divided in adjacent segments of 100Kb (per chromosome); each plot shows the results for all segments in the chromosome (with ratio of qualified positions $>$ 0.9).'
plots.texForOnePlot_AllChromosomes(dotInFileNames_b = 0, chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart,  plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)




#droso
chromosomeList = ['r6.18_chrX', 'r6.18_chr2L', 'r6.18_chr2R', 'r6.18_chr3L', 'r6.18_chr3R','r6.18_chr4']
chromosomeDict = {'r6.18_chrX':22, 'r6.18_chr2L':32, 'r6.18_chr2R':24, 'r6.18_chr3L':27, 'r6.18_chr3R':31,'r6.18_chr4':1 }

#chromosomeList = ['r6.18_chr3R']

#generate sub-string list:
subListDict = plots.genSubListDict(chromosomeBaseDict = chromosomeDict, subListType = 0)
#gen text:
rootFileNamePart = ['Fourier_norm_w1000_step100_',  '_seg1000000_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForDrosoLSTM4FouriersAllChromos.txt'
caption = ['Drosophila LSTM4. Fouriers on reference-base probability for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
plots.texForMultiPlot_ChromosomeBySub(dotInFileNames_b = 1, chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart, subListDict = subListDict, plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)


#One plot of all the all-segments plots (ie all chromos)
#gen text:
rootFileNamePart = ['Fourier_norm_w5000_step100_', '_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForDrosoLSTM4FouriersAllChromosTotOnePlot_hiFrqs.txt'
caption = 'Drosophila LSTM4. Fouriers on reference-base probability, higher frequency range. Each plot covers one chromosome, listed in increasing order (chrX, chr2L, chr2R, chr3L, chr3R, chr4). The genome string is divided in adjacent segments of 1Mb (per chromosome); each plot shows the results for all segments in the chromosome (with ratio of qualified positions $>$ 0.9).'
plots.texForOnePlot_AllChromosomes(dotInFileNames_b = 1, chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart,  plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)




#zebrafish
chromosomeList = ['GRCz11_chr1', 'GRCz11_chr2', 'GRCz11_chr3', 'GRCz11_chr4', 'GRCz11_chr5', 'GRCz11_chr6', 'GRCz11_chr7', 'GRCz11_chr8','GRCz11_chr9', 'GRCz11_chr10', 'GRCz11_chr11', 'GRCz11_chr12','GRCz11_chr13', 'GRCz11_chr14', 'GRCz11_chr15', 'GRCz11_chr16', 'GRCz11_chr17', 'GRCz11_chr18','GRCz11_chr19', 'GRCz11_chr20', 'GRCz11_chr21', 'GRCz11_chr22','GRCz11_chr23', 'GRCz11_chr24', 'GRCz11_chr25']
chromosomeDict = {'GRCz11_chr1':58, 'GRCz11_chr2':58, 'GRCz11_chr3':61, 'GRCz11_chr4':77, 'GRCz11_chr5':71, 'GRCz11_chr6':60, 'GRCz11_chr7':73, 'GRCz11_chr8':53,'GRCz11_chr9':55, 'GRCz11_chr10':44, 'GRCz11_chr11':44, 'GRCz11_chr12':48,'GRCz11_chr13':51, 'GRCz11_chr14':51, 'GRCz11_chr15':47, 'GRCz11_chr16':54, 'GRCz11_chr17':52, 'GRCz11_chr18':50,'GRCz11_chr19':47, 'GRCz11_chr20':54, 'GRCz11_chr21':44, 'GRCz11_chr22':38,'GRCz11_chr23':45, 'GRCz11_chr24':41, 'GRCz11_chr25':36}

#chromosomeList = ['GRCz11_chr7']

#generate sub-string list:
subListDict = plots.genSubListDict(chromosomeBaseDict = chromosomeDict, subListType = 0)
#gen text:
rootFileNamePart = ['Fourier_norm_w1000_step100_',  '_seg1000000_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForZebrafishLSTM4FouriersAllChromos.txt'
caption = ['Zebrafish LSTM4. Fouriers on reference-base probability for chromosome: ', '. The genome string is divided in adjacent segments of 1Mb; several adjacent segments are covered in each plot, segment numbers are indicated in the legend. First plot (upper left) shows the results for all segments in this chromosome.']
plots.texForMultiPlot_ChromosomeBySub(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart, subListDict = subListDict, plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)

#One plot of all the all-segments plots (ie all chromos)
#gen text:
rootFileNamePart = ['Fourier_norm_w5000_step100_', '_', 'avgRevCompl0_win1_step1']
plotsPerRow = 5
rootOutput = 'C:/Users/Christian/Sync/Bioinformatics/papers/DNAproject/plots/'
fileNameOut = 'texForZebrafishGCbiasFouriersAllChromosTotOnePlot_hiFrqs.txt'
caption = 'Zebrafish LSTM4. Fouriers on GC-content. Each plot covers one chromosome, listed in increasing order (chr1 to chr25). The genome string is divided in adjacent segments of 1Mb (per chromosome); each plot shows the results for all segments in the chromosome (with ratio of qualified positions > 0.9).'
plots.texForOnePlot_AllChromosomes(chromosomeList = chromosomeList, rootFileNamePart = rootFileNamePart,  plotsPerRow = plotsPerRow, caption = caption, rootOutput = rootOutput , fileNameOut = fileNameOut, widthPlots = 0.18)


###########################################################################################################


'''

#import cPickle as pickle
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np


from math import factorial

modelNameChangeDict = {'LSTM1':'LSTM200', 'LSTM4':'LSTM50','LSTM4P':'LSTM50P', 'LSTM4Q':'LSTM50Q', 'LSTM4R':'LSTM50R', 'LSTM4S':'LSTM50S', 'LSTM11':'LSTM200early', 'mouseLSTM4':'mouseLSTM50'}

colorMap='Set2'
colorList = cm.get_cmap(colorMap)
colorDict = {'LSTM1':colorList(0), 'LSTM4S':colorList(1), 'LSTM4':colorList(2), 'LSTM4P':colorList(3), 'LSTM11':colorList(4), 'mouseLSTM4':colorList(6), 'LSTM4R':colorList(6), 'LSTM4Q':colorList(7) }


def collectivePerfPlot(modelFileNameList, batchSizeRatioList, lastRepeatNrList, epochsPerRepeatList, modelNameList, rootOutput, bigLoopIterNr = 0, fileNameAcc = 'collectivePlot_total_trainTest_acc_vs_allEpochs', fileNameLoss = 'collectivePlot_total_trainTest_loss_vs_allEpochs', colorDict = colorDict,  fontSizeLabels = 'medium'):
    '''For making plots of performance (acc and loss) for several models across
    their complete training schedule'''
    
    accTotalHistDictTrain = {}
    lossTotalHistDictTrain = {}
    accTotalHistDictTest = {}
    lossTotalHistDictTest = {}
    accTotalHistDictTestInterp = {}
    lossTotalHistDictTestInterp = {}
    batchSizeRatioDict = {}
    
        
    
    cnt = 0
    for modelFileName in modelFileNameList:
        
        accTotalHistDictTrain[modelFileName] = []
        lossTotalHistDictTrain[modelFileName] = []
        
        accTotalHistDictTest[modelFileName] = []
        lossTotalHistDictTest[modelFileName] = []
                
        batchSizeRatioDict[modelFileName] = batchSizeRatioList[cnt]
        
        
        #load history of acc and loss:
        for i in range(lastRepeatNrList[cnt]+1): #+1: else we exclude the training results for lastRepeatNrList[cnt]
            
            #for the training part
            loadFile = modelFileName + '_bigLoopIter' + str(bigLoopIterNr) + '_repeatNr' + str(i) + '_training_loss_vs_epoch.p'
            lossThisRepeat = pickle.load(open( loadFile, "rb"))
#            if len(lossThisRepeat) != 100:
#                print "len loss ", len(lossThisRepeat) 
            lossTotalHistDictTrain[modelFileName].extend(lossThisRepeat)
            
            loadFile = modelFileName + '_bigLoopIter' + str(bigLoopIterNr) + '_repeatNr' + str(i) + '_training_acc_vs_epoch.p'
            accThisRepeat = pickle.load(open( loadFile, "rb"))
#            if len(accThisRepeat) != 100:
#                print "len acc ", len(accThisRepeat) 
            accTotalHistDictTrain[modelFileName].extend(accThisRepeat)
                                    
        #for the testing part
        j = lastRepeatNrList[cnt]
        loadFile = modelFileName + '_bigLoopIter' + str(bigLoopIterNr) + '_repeatNr' + str(j) + '_testing_loss_vs_epoch.p'
        lossThisRepeat = pickle.load(open( loadFile, "rb"))
        lossTotalHistDictTest[modelFileName].extend(lossThisRepeat)
        
        loadFile = modelFileName + '_bigLoopIter' + str(bigLoopIterNr) + '_repeatNr' + str(j) + '_testing_acc_vs_epoch.p'
        accThisRepeat = pickle.load(open( loadFile, "rb"))
        accTotalHistDictTest[modelFileName].extend(accThisRepeat)
        
        cnt += 1
        
    #The test results are had in a list stating the outcome of the test at each repeat ie once every epochsPerRepeat epochs; the training results
    #are had once per epoch; so to plot the two toghether we linearly interpolate between the test results had:
    cnt = 0
    toFill = {}
    for modelFileName in modelFileNameList:
        
        #the tests are had at the end of each repeat; hence the first test is had a epochsPerRepeat; we therefore put the 
        #values on that range of x-values:
        batchRatio = batchSizeRatioDict[modelFileName]
        toFill[modelFileName] = np.arange(start = batchRatio*epochsPerRepeatList[cnt], stop = batchRatio*(lastRepeatNrList[cnt]+1)*epochsPerRepeatList[cnt] + 1) #last is then batchRatio*(lastRepeatNrList[cnt]+1)*epochsPerRepeatList[cnt]
        haveTheseXs = np.arange(start = batchRatio*epochsPerRepeatList[cnt], stop = batchRatio*(lastRepeatNrList[cnt]+2)*epochsPerRepeatList[cnt], step = batchRatio*epochsPerRepeatList[cnt] ) #last is then batchRatio*(lastRepeatNrList[cnt]+1)*epochsPerRepeatList[cnt]
        haveTheseYs = lossTotalHistDictTest[modelFileName]
        #print haveTheseXs
        #print len(haveTheseXs), len(haveTheseYs)
        lossTotalHistDictTestInterp[modelFileName] = np.interp(toFill[modelFileName], haveTheseXs, haveTheseYs)
        haveTheseYs = accTotalHistDictTest[modelFileName]
        accTotalHistDictTestInterp[modelFileName] = np.interp(toFill[modelFileName], haveTheseXs, haveTheseYs)
        cnt += 1
        
    
    fig = plt.figure()
    cnt = 0
    for modelKey in modelFileNameList:
        
        modelName = modelNameList[cnt]
        modelNameForLegend = modelNameChangeDict[modelName]
        
        if not(modelKey in accTotalHistDictTrain):
            
            print("Key ", modelKey, " not known")
            return
    
        # plot history for accuracy
        batchRatio = batchSizeRatioDict[modelKey]
        xRange = np.arange(batchRatio*1, batchRatio*(len(accTotalHistDictTrain[modelKey])+1), step = batchRatio) #first epoch = 1
        #print(xRange, toFill[modelKey])
        plt.plot(xRange, accTotalHistDictTrain[modelKey] , color = colorDict[modelNameList[cnt]], alpha = 0.25)
        plt.plot(toFill[modelKey], accTotalHistDictTestInterp[modelKey], label = modelNameForLegend , color = colorDict[modelNameList[cnt]])
        plt.title('Model accuracy, training and validation')
        plt.xticks(fontsize = fontSizeLabels)
        plt.yticks(fontsize = fontSizeLabels)
        plt.ylabel('accuracy', fontsize = fontSizeLabels)
        plt.xlabel('epoch', fontsize = fontSizeLabels)
#        plt.yticks(np.arange(0.35, 0.55, step = 0.01))
        plt.legend(loc='best', fontsize = fontSizeLabels)

        cnt += 1 
        
    plt.savefig(rootOutput + fileNameAcc + '.pdf')
#    plt.show()
    plt.close()
        
        
    fig = plt.figure()
    cnt = 0
    for modelKey in modelFileNameList:
        
        modelName = modelNameList[cnt]
        modelNameForLegend = modelNameChangeDict[modelName]

        
        if not(modelKey in lossTotalHistDictTrain):
            
            print("Key ", modelKey, " not known")
            return
    
         # plot history for loss
        batchRatio = batchSizeRatioDict[modelKey]
        xRange = np.arange(batchRatio*1, batchRatio*(len(lossTotalHistDictTrain[modelKey])+1), step = batchRatio) #first epoch = 1
        plt.plot(xRange,lossTotalHistDictTrain[modelKey], color =colorDict[modelNameList[cnt]], alpha = 0.25)
        plt.plot(toFill[modelKey],lossTotalHistDictTestInterp[modelKey], label = modelNameForLegend, color = colorDict[modelNameList[cnt]])
        plt.title('Model loss, training and validation')
        plt.xticks(fontsize = fontSizeLabels)
        plt.yticks(fontsize = fontSizeLabels)
        plt.ylabel('loss', fontsize = fontSizeLabels)
        plt.xlabel('epoch', fontsize = fontSizeLabels)
        plt.legend(loc='best', fontsize = fontSizeLabels)
#        plt.yticks(np.arange(1.0, 1.35, step = 0.01))
        cnt += 1
        
    plt.savefig(rootOutput +  fileNameLoss + '.pdf')
#    plt.show()
    plt.close()     
    


def multimodelTrainTestFigures(rootOutputList, modelNameList, modelFileNameList, modelNameChangeDict, repeatNrList, accDictFileNameList, sampleSizeList, sampleFactorList, rootOutputPlot, saveAtDpi = 300, usePartition ='part1', offsetValue = 0.0, useCmap = 'tab20', plot_b = True, fontSizeLabels = 'medium', legend_b = 1):
    '''   repeatNrList: to contain, for each of the models in modelNameList, the repeat number at which the trained model was taken out. (This
    need not be the last of a very late repeat number, but if not, the estimated accuracy on not-trained-on samples is not neccesaarily con-
    servative).''' 
    
    modelByEstimateDict = {}
    estimateByModelDict = {}
    samplingInfoByModelDict = {}
    
    modelByEstimateDict['acc all'] = {}
    modelByEstimateDict['acc estimate'] = {}
    modelByEstimateDict['acc last train'] = {}
    modelByEstimateDict['acc last val'] = {}
    
    estimatesList = modelByEstimateDict.keys()
    print(estimatesList)
    
    nModels = len(modelNameList)
    for i in range(nModels):

        rootOutput = rootOutputList[i]
        modelName = modelNameList[i]
        modelFileName = modelFileNameList[i]
        repeatNr = repeatNrList[i]
        accDictFileName = accDictFileNameList[i]
         
        #load training and validation results for last repeat ('epoch'):
        loadFileTrain = rootOutput + modelName + modelFileName  + str(repeatNr) + '_training_acc_vs_epoch.p'
        trainAccThisRepeat = pickle.load(open( loadFileTrain, "rb"))
        
        #let the 'training accuracy' be rep'ed by the average across the last repeat:
        lastTrainAvg = np.mean(trainAccThisRepeat)
        lastTrainStd = np.std(trainAccThisRepeat)
         
        loadFileTest = rootOutput + modelName + modelFileName + str(repeatNr) + '_testing_acc_vs_epoch.p'
        testAccThisRepeat = pickle.load(open( loadFileTest, "rb"))
         
        #the 'validation acc' is taken to be the validation result in the last repeat: 
        testAcc = testAccThisRepeat[::-1][0]
         
        accDictFile = rootOutput + modelName + accDictFileName
        accDict = pickle.load(open( accDictFile, "rb"))


        if modelName == 'LSTM4S':
            accDict, partitionInfo = accDict
            accAll_ = accDict[usePartition]['all']
        else:
            print(accDict)
            accAll_ = accDict['all'] #structure: accuracy whole genome, n of true predictions, n of positions/contexts in whole genome predicted

        #accAll_ = accDict['all'] #structure: accuracy whole genome, n of true predictions, n of positions/contexts in whole genome predicted
            
        accAll = accAll_[0]
        
        nSamplesPerRound = sampleSizeList[i]
        nSamples = nSamplesPerRound*(repeatNr +1 ) #first repeat is nr 0, so must add 1
        sampleFactor = sampleFactorList[i]
        Ntotal =  accAll_[2] #if trained on 'whole' genome this should be close to the genome size in bps (single strand)
        #for the LSTM4P model accAll_[2] will contain the number of positions on which the prediction was done; so
        #we have to change this to the complementary to get the #training samples:
        if modelName == 'LSTM4P':
            Ntotal = 2747445252 - Ntotal #2747445252 is taken from LSTM4/LSTM1
        
        #compute the expected number of different samples obtained by sampling n times from the genome of size N:
        NtotalTrain = Ntotal*sampleFactor #total nr of positions in the sampling for the training 
        nrOfDiffSamples = expectedNrOfDiffSamples(n = nSamples, N = NtotalTrain, tolerance = 0.0001)
        print("Nr of diff samples is estmated to ", nrOfDiffSamples)
        
        #to estimate the the accuracy on the non-trained part, we make th conservative assumption that
        #the average accuracy obtained in the last training 'round' is valid for all samples covered
        #in the training -- about nrOfDiffSamples in total. Then the desired estimate fits
        #acc_all = fraction-of-trained-samples*avg_acc_training + (1 - fraction-of-trained-samples)*estimate
        #Which amounts to:
        accEstimate = (accAll - float(nrOfDiffSamples)/Ntotal*lastTrainAvg)/(1 - float(nrOfDiffSamples)/Ntotal)
        accEstimate_err = (float(nrOfDiffSamples)/Ntotal*lastTrainStd)/(1 - float(nrOfDiffSamples)/Ntotal)
        
        print(accAll, accEstimate, lastTrainAvg, lastTrainStd, testAcc)
        
        modelByEstimateDict['acc all'][modelName] = accAll
        modelByEstimateDict['acc estimate'][modelName] = accEstimate, accEstimate_err
        modelByEstimateDict['acc last train'][modelName] = lastTrainAvg, lastTrainStd
        modelByEstimateDict['acc last val'][modelName] = testAcc        

        estimateByModelDict[modelName] = {}
        estimateByModelDict[modelName]['acc all'] =  accAll
        estimateByModelDict[modelName]['acc estimate'] = accEstimate
        estimateByModelDict[modelName]['acc last train'] = lastTrainAvg, lastTrainStd
        estimateByModelDict[modelName]['acc last val'] = testAcc
        
        samplingInfoByModelDict[modelName] = {}
        samplingInfoByModelDict[modelName]['nTotal'] = Ntotal 
        samplingInfoByModelDict[modelName]['nTotalTrain'] = int(NtotalTrain)
        samplingInfoByModelDict[modelName]['nrRepeats'] = int(repeatNr + 1)
        samplingInfoByModelDict[modelName]['nSamplesPerRepeat'] = int(nSamplesPerRound)
        samplingInfoByModelDict[modelName]['nrSamples'] = int(nSamples)
        samplingInfoByModelDict[modelName]['expectedNrDiffSamples'] = int(round(nrOfDiffSamples,0))
        samplingInfoByModelDict[modelName]['expectedCoverage'] = round(float(nSamples)/nrOfDiffSamples,2)



    nEstimates = len(estimatesList)
    print("nEstimates ", nEstimates)


    if plot_b:
        colors = cm.get_cmap(useCmap)
        
        X = np.arange(nModels)
        bar_width = 1./(nEstimates +1)
    
        fig, ax = plt.subplots()
    #        ax = fig.add_axes([0,0,1.2,1.2])
        
    #        print dataDict
        
        cnt = 0
        for estimateName in estimatesList:
            
            if estimateName == 'acc last train' or estimateName == 'acc estimate':            
                plotThis = [modelByEstimateDict[estimateName][modName][0] - offsetValue for modName in modelNameList]
                plotThis_errBar = [modelByEstimateDict[estimateName][modName][1] for modName in modelNameList]
                
                ax.bar(X + cnt*bar_width , plotThis,  yerr=plotThis_errBar,  color = colors(cnt), width = bar_width, label = estimateName)
                #ax.errorbar(X + cnt*bar_width, plotThis, yerr=plotThis_errBar, color="black")
                
            else:
                plotThis = [modelByEstimateDict[estimateName][modName] - offsetValue for modName in modelNameList]
        
                ax.bar(X + cnt*bar_width , plotThis, color = colors(cnt), width = bar_width, label = estimateName)
            
            cnt += 1
    
        #set the labels of the yticks right, corr to the offset:
        plt.yticks(fontsize = fontSizeLabels)
        plt.draw()
        yTicks = ax.get_yticks()
        yTickLabels = [item.get_text() for item in ax.get_yticklabels()]
        #print yTickLabels
        yTickLabelsNew = [str(float(yLbl) + offsetValue)  for yLbl in yTickLabels]
        
        plt.yticks(yTicks,yTickLabelsNew, fontsize = fontSizeLabels)
        ax.set_ylabel('Accuracy', fontsize = fontSizeLabels)
    #        ax.set_title('...')
    #        ax.set_xticks(X + bar_width / 2)
        if modelNameChangeDict:
            useTheseModelNames = [modelNameChangeDict[modName] for modName in modelNameList]
        else:
            useTheseModelNames = [modelName in modelNameList]
            
        plt.xticks(X + (nEstimates-1)*0.5*bar_width,useTheseModelNames, rotation = 90, fontsize = fontSizeLabels)
        ax.set_xlabel('Model', fontsize = fontSizeLabels)
        if legend_b == 1:
            ax.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', fontsize = fontSizeLabels) #places legend outside the frame with the upper left placed at coords (1.05, 0.6)
        
        plt.tight_layout()
        #plt.show()
            
        plt.savefig(rootOutputPlot + 'estimates_by_model.pdf', dpi=saveAtDpi)
    
        plt.close() 
    
    #dump the result
    dumpFile = rootOutput +'estimateByModelDict'
    pickle.dump(estimateByModelDict, open(dumpFile, "wb") )
    
    return estimateByModelDict, samplingInfoByModelDict


#This is id to the same fct in dnaNet_stats:
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
        if nameChangeDict.has_key(colKey):
            colName = nameChangeDict[colKey]
        
        subS += r'& \textbf{' + colName + '}'
        
    subS += r'\\'   + '\n'
    
    s += subS    
    
    s +=r'      \hline' + '\n'
    
    for rowKey in rowKeys:
        
        subS = rowKey
        #replace if wanted:
        if nameChangeDict.has_key(rowKey):
            subS = nameChangeDict[rowKey]
  
    
        for colKey in colKeys:
            
            if not(inputDict[rowKey].has_key(colKey)):
                
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
    
    


def texForMultiPlotByChromosome(chromosomeList, rootFileNamesList, caption, rootOutput, fileNameOut, widthPlots = 0.2):
    
    plotsPerRow = len(rootFileNamesList)
    
    texString = r'\begin{figure}' + '\n'
    texString += r'\centering' + '\n'
    nrPlots = 0
    for chromo in chromosomeList:
        
        for nrPlots in range(plotsPerRow):
            
            stringFile = rootFileNamesList[nrPlots] + chromo + '.pdf'
            
            texString += r'\begin{minipage}[t]{' + str(widthPlots) + r'\textwidth}' + '\n'
            texString += r'\includegraphics[width=\textwidth]{' + stringFile + '}' + '\n'
            texString += r'\end{minipage}' + '\n'
                        
        texString += '\n'
        
    texString += r'\caption{' + caption + '}'  + '\n'
    texString += r'\end{figure}' + '\n'
    
    outputFileName = rootOutput + fileNameOut
    outputFile = open(outputFileName, 'w')
    outputFile.write(texString)
    outputFile.close()
    
    return texString
    

def genSubListDict(chromosomeBaseDict, subListType = 0, step = 10):

    subListDict = {}    
    if subListType == 0:
        
        for chromo in chromosomeBaseDict:
            
            subListDict[chromo] = []
            
            maxIdx = chromosomeBaseDict[chromo]
            
            idx = 0
            while idx + step -1 < maxIdx:
            
                subListDict[chromo].append('seg' + str(idx) + '_' + str(idx + step -1) + '_')
                
                idx += step
            
            if idx == maxIdx :
                subListDict[chromo].append('seg' + str(idx) + '_' + str(idx -1 ) + '_')
            else:
                subListDict[chromo].append('seg' + str(idx) + '_' + str(maxIdx) + '_')
                
            
    return subListDict
    
    
def texForMultiPlot_ChromosomeBySub(chromosomeList, rootFileNamePart, subListDict, plotsPerRow, caption, rootOutput, fileNameOut, widthPlots = 0.2, dotInFileNames_b = 0):
    '''rootFileNamePart: list of three parts, prefix, middle and post. Eg
    Fourier_norm_w1000_step100_,  _seg1000000_, avgRevCompl0_win1_step1'''
    
    texString = ''
    for chromo in chromosomeList:
        
        texString += r'\begin{figure}' + '\n'
        texString += r'\centering' + '\n'
        
        #if >1 elt in rootFileNamesList, first plot, first row is 1st elt in rootFileNamesList
        if dotInFileNames_b == 1:
            stringFile = '{' + rootFileNamePart[0] + chromo + rootFileNamePart[1] + rootFileNamePart[2] + '}.pdf'
        else:
            stringFile = rootFileNamePart[0] + chromo + rootFileNamePart[1] + rootFileNamePart[2] + '.pdf'
        
        texString += r'\begin{minipage}[t]{' + str(widthPlots) + r'\textwidth}' + '\n'
        texString += r'\includegraphics[width=\textwidth]{' + stringFile + '}' + '\n'
        texString += r'\end{minipage}' + '\n'
        
        rowNr = 0
        nrPlots = 1 #since we've already placed one
    
        for subStr in subListDict[chromo]:
            
#            print subStr
            
#            if rowNr == 0:
#                startAt = 1
#            else:
#                startAt = 0
            
#            for nrPlots in range(startAt, plotsPerRow):
            if dotInFileNames_b == 1:    
                stringFile = '{' + rootFileNamePart[0] + chromo + rootFileNamePart[1] + subStr + rootFileNamePart[2] + '}.pdf'
            else:
                stringFile = rootFileNamePart[0] + chromo + rootFileNamePart[1] + subStr + rootFileNamePart[2] + '.pdf'
                
            texString += r'\begin{minipage}[t]{' + str(widthPlots) + r'\textwidth}' + '\n'
            texString += r'\includegraphics[width=\textwidth]{' + stringFile + '}' + '\n'
            texString += r'\end{minipage}' + '\n'
                 
            if nrPlots == plotsPerRow-1:
                texString += '\n'
                nrPlots = 0
            else:
                nrPlots += 1
            
            rowNr += 1        
    
        texString += r'\caption{' + caption[0] + ' ' + chromo + caption[1] + '}'  + '\n'
        texString += r'\end{figure}' + '\n'
        
        texString += '\n'
        texString += '\n'
        texString += '\n'
        texString += '\n'
    
    outputFileName = rootOutput + fileNameOut
    outputFile = open(outputFileName, 'w')
    outputFile.write(texString)
    outputFile.close()
    
    return texString
        
        

def texForOnePlot_AllChromosomes(chromosomeList, rootFileNamePart, plotsPerRow, caption, rootOutput, fileNameOut, widthPlots = 0.2, dotInFileNames_b = 0):
    '''rootFileNamePart: list of three parts, prefix, middle and post. Eg
    Fourier_norm_w1000_step100_,  _seg1000000_, avgRevCompl0_win1_step1'''
    
    texString = ''
        
    texString += r'\begin{figure}[H]' + '\n'
    texString += r'\centering' + '\n'
    
    nrPlots = 0 
    for chromo in chromosomeList: 
        
        if dotInFileNames_b == 1:
            stringFile = '{' + rootFileNamePart[0] + chromo + rootFileNamePart[1] + rootFileNamePart[2] + '}.pdf'
        else:
            stringFile = rootFileNamePart[0] + chromo + rootFileNamePart[1] + rootFileNamePart[2] + '.pdf'
        
        texString += r'\begin{minipage}[t]{' + str(widthPlots) + r'\textwidth}' + '\n'
        texString += r'\includegraphics[width=\textwidth]{' + stringFile + '}' + '\n'
        texString += r'\end{minipage}' + '\n'
                         
        if nrPlots == plotsPerRow-1:
            texString += '\n'
            nrPlots = 0
        else:
            nrPlots += 1
            
    
    texString += r'\caption{' + caption + '}'  + '\n'
    texString += r'\end{figure}' + '\n'
    
    texString += '\n'
    texString += '\n'
    texString += '\n'
    texString += '\n'
    
    outputFileName = rootOutput + fileNameOut
    outputFile = open(outputFileName, 'w')
    outputFile.write(texString)
    outputFile.close()
    
    return texString
        
    

def ln_k_of_n(n, k):
    
    

    #The number of times a given object is drawn is binomially distr'ed, with p_success = p. Using
    #Stirling's formula for approximating when k is above 100 we get the k-of-n number of possible draws
    #(for k less than 100 we just compute directly):

    if k < 100:
        
        #we must calculate n*(n-1)* ... *(n-k+1)/k!
        k_of_n = 1./factorial(k)
        for j in range(k):
              k_of_n = k_of_n*(n-j)
              
        ln_k_of_n = np.log(k_of_n)

    else:
        
        ln_k_of_n =  - np.log(np.sqrt(2*np.pi*k)) - k*np.log(k/np.e) + k*np.log(n/np.e) + (0.5 + n - k)*np.log(float(n)/(n-k))
    
    return ln_k_of_n


def expectedNrOfDiffSamples(n, N, tolerance):
    ''' Returns the expected number of different samples when sampling n times from a pool of 
    N objects, all equally likely. 
    
    tolerance: with N very large (size of genome) the probability of getting drawn is very small; for some k the likelihood
    of getting drawn k times is so close to zero that we can assume that no object is sampled more than k times; 
    the contribution to the sought expectation is the prob of getting exactly k draws times the number of samples drawn;
    contributions smaller than the set tolerance are discarded.
    
    (was tested by initialising k to zero, below)
    '''
    
    #prob of success, ie of sampling a given object
    p = 1./N    

    #The number of times a given object is drawn is binomially distr'ed, with p_success = p. Using
    #Striling's formula for approximating m! (large m) we get the desired expectation by summing
    #up all contributions smaller than the tolerance:
    contribution_k_1 = 0
    contribution_k = 0
    desiredExpectation = 0
    accum_prob = 0
    k = 1 #replace this by zero to test this function ...
    while (contribution_k >= contribution_k_1 or contribution_k > tolerance):
    
        print(n, k, ln_k_of_n(n, k))
        ln_prob_exactly_k = ln_k_of_n(n, k) + k*np.log(p) + (n-k)*np.log(1 - p)
        
        prob_exactly_k = np.exp(ln_prob_exactly_k)
        print(prob_exactly_k)        
        
        contribution_k_1 = contribution_k #reset
        contribution_k = prob_exactly_k*N
        
        desiredExpectation += contribution_k
        
        accum_prob += prob_exactly_k

        k += 1 
        
    print(accum_prob)
        
    return desiredExpectation

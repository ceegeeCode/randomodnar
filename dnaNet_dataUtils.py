# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:04:04 2020

@author: Christian Grønbæk

"""

'''
The code below -- all in a 'research state' -- contains a few data utils used in our work with the neural networks 
as reported in the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 

Notes:

    -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
    -- this module is very small and only used in few places (and really no direct use in the paper above)


'''
import dnaNet_dataUtils as dataUtils

#Human

#LSTM4 (LSTM5 really -- has dense50 rather than dense20): flanks 50, trained on hg38, w train test split 80/20
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!

#GC bias
#this is to get the qual arrays from the model-pred run:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/GCbias/notAvgRevCompl/"
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromoName = 'hg38_chr20'



#Mouse

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'

#placeholders
modelFileName_forATorGCbias = ''
rootOutput_forATorGCbias = ''
forATorGCbias_b = 0 #!!!!!!!!!!!!


#GC bias
#this is to get the qual arrays from the model-pred run:
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
#this is to get the predReturs fo the bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/GCbias/notAvgRevCompl/"
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromoName = 'm38_chr11'




#General settings:

segmentLength = 1000000

augmentWithRevComplementary_b = 0  #!!!!!

#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
                     
shuffle_b = 0 #!!!!!!!!!!!!!!!!!!!!

#Which input to use:
inputArrayType = 0 #1: ref base prob's; 0: pred returns

nrSegments = 500

rootInput = rootOutput + chromoName + r'/'
rootInput_forATorGCbias = rootOutput_forATorGCbias + chromoName + r'/'

dataUtils.writeSomeArraysToFile(rootInput = rootInput,
                             modelFileName = modelFileNameNN,  
                             segmentLength = segmentLength,
                             inputArrayType = inputArrayType,
                             averageRevComplementary_b = augmentWithRevComplementary_b,
                             chromoName = chromoName,
                             nrSegments = nrSegments,
                             windowLength = windowLength,
                             stepSize = stepSize,
                             shuffle_b = shuffle_b,
                             forATorGCbias_b = forATorGCbias_b, 
                             rootInput_forATorGCbias= rootInput_forATorGCbias,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias)



'''

import cPickle as pickle

import pandas as pd

import numpy as np

import dnaNet_dataGen as dataGen



def writeSomeArraysToFile(rootInput, 
                             modelFileName, 
                             segmentLength,
                             chromoName,
                             averageRevComplementary_b,
                             inputArrayType = 0,
                             windowLength = 1,
                             stepSize = 1,
                             nrSegments = 1000, #with 1Mb segments, this will suffice for any single chromo ...
                             shuffle_b = 0,
                             forATorGCbias_b = 0,
                             rootInput_forATorGCbias = '',
                             modelFileName_forATorGCbias = ''):                   
                             
    
    genomeIdName = chromoName + '_seg' + str(int(segmentLength))      
  
    if shuffle_b == 1:
        raw_input("I'll shuffle the input arrays!")
    
    for i in range(nrSegments):
        
        genomeIdNameSeg = genomeIdName + '_segment' + str(i)
        if forATorGCbias_b == 0:
            
            if inputArrayType == 0:
                loadFile_pred = rootInput +  modelFileName + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   

            elif inputArrayType == 1:
                loadFile_pred = rootInput +  modelFileName + '_predArray_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) 
                
        else:
            
            loadFile_pred = rootInput_forATorGCbias +  modelFileName_forATorGCbias + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
        
#        loadFile = rootOutput + modelFileName + '_' + 'predArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)        
        
        print loadFile_pred
        

        try:
            
            if inputArrayType == 0:
                
                storedPredThisSeg = pickle.load(open(loadFile_pred,"rb"))
                
                avgPredSeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = storedPredThisSeg

    #            avgPredSeg = storedPredThisSeg
                print i, avgPredSeg.shape
    #            print avgPredSeg[100000:101000]
    #            raw_input("S er den dejligste")
            elif inputArrayType == 1: #in this case we want the array to contain the probability of the ref base assigned by the model
                
                #the prob of the ref base is obtained simply by taking the dot-product of 
                #the predArray and the label array, which we therefore also load:
                predArrayThisSeg = pickle.load(open(loadFile_pred,"rb"))
                
                loadFile_label = rootInput + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                labelArrayThisSeg = pickle.load(open(loadFile_label,"rb"))
                
                #take the entry-wise dot product; we use avgPredSeg as name for convenience (since used for predReturn/inputArrayType=0-case ) 
                L = labelArrayThisSeg.shape[0]
                avgPredSeg = np.zeros(shape = L, dtype = 'float64')
                for l in range(L):
                    avgPredSeg[l] = np.dot(predArrayThisSeg[l],labelArrayThisSeg[l])
                                    
            
            #get the qualified array too:
            loadFile_qual = rootInput + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
            qualArraySeg = pickle.load(open( loadFile_qual, "rb" ) )
            print qualArraySeg.size
            
            #shuffle the two arrays if desired:
            if shuffle_b == 1:
                
                shuffledIdxs = np.arange(avgPredSeg.shape[0])
                np.random.shuffle(shuffledIdxs)
                
                avgPredSeg = avgPredSeg.take(shuffledIdxs)
                qualArraySeg = qualArraySeg.take(shuffledIdxs)
                
            
            np.savetxt(loadFile_pred + '_.csv', avgPredSeg, delimiter=",")
            np.savetxt(loadFile_qual + '_.csv', qualArraySeg, delimiter=",")   
            
                
        except IOError:
            print "Files not found, last tried: ", loadFile_pred
            return
        except EOFError:
            print "Empty file, last tried: ", loadFile_pred
            return
            
            
            
            
def getFullGCcontentArray(GCcontentArray, labelArray, qualArray):
    ''' 
    When generating the GC-content arrays in connection to a model, we
    set the GC content to some default value at all non-qualified positions; so since for 
    each case of non-acgtACGT letter the whole context around it is disqualified, all
    these positions will be treated as having the default GC-content.
    
    For broader use of the GC-content arrays it is desirable to keep the actual
    GC/AT value at all positions holding an acgtACGT, and only use the default
    at non-acgtACGT letters.
    
    This fct returns such a full GC-content array based on the needed input.
    '''
    
    #Simply run through all disqualified positions and update the GC-array
    #based on the labelArray content:
    disQidxs = np.where(qualArray == 0)[0]
    
    print "Nr disQ: ", len(disQidxs) 
    
    GCcontentArrayOut = GCcontentArray.copy()
    
    print "Sum GC, before: ", np.sum(GCcontentArray)
    
    cntAT = 0
                
    for idx in disQidxs:
        
        if np.array_equal(labelArray[idx], dataGen.codeW_asArray):
            
            continue
        
        elif np.array_equal(labelArray[idx], dataGen.codeA_asArray):
            
            GCcontentArrayOut[idx] = 0
            cntAT +=1
        
        elif np.array_equal(labelArray[idx], dataGen.codeG_asArray):
            
            GCcontentArrayOut[idx] = 1             
        
        elif np.array_equal(labelArray[idx], dataGen.codeC_asArray):
            
            GCcontentArrayOut[idx] = 1
        
        elif np.array_equal(labelArray[idx], dataGen.codeT_asArray):
            
            GCcontentArrayOut[idx] = 0
            cntAT += 1 
            
    print "Sum GC, after: ", np.sum(GCcontentArrayOut)
    print "cntAT at disQs: ", cntAT
#    raw_input("S..... .. ........")
            
    return GCcontentArrayOut
            
            
                    
                
                
                
        
       
        

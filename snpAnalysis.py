 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:11:30 2019

@author: Christian Grønbæk

"""

'''
The code below -- all in a 'research state' -- was used for the analysis of single nucleotide polymorphims (SNPs) 
as reported in the following paper (the SNP analysis is placed in a suppplementary to the paper):

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 



Notes:

    -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
    -- the example below take for grated that the SNP data have been downloaded and the file containing this data
        matches the code below (readSNPdata)
    -- more calls of the functions in this module are placed in the Usage section part 8.1 of the dnaNet_stats module 
      (which is where the plots etc for the paper's Supplementary on SNPs are made) 

##################################################################################################
# Usage:
##################################################################################################

The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.

    
import snpAnalysis as snp


#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'

chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'

snpData = snp.readSNPdata(rootSnp + fileName)

#For usage of the fetchProbsAtSnps fct see the dnaNet_stats module's Usage section 8.1

'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm

import numpy as np

import dnaNet_dataGen as dataGen

import csv

import os

import cPickle as pickle


def readSNPdata(snpDataBaseFile):
    
    '''Read in a file containing: chromo-id, position in chromo, ref base, alt base.
    
    Returns: array of (chromo nr, postion, ref base, alt base) where the bases are 1-hot encoded 
    (as in dnaNet_datagen; letter order ACGT)'''
    
    infoList = []
    
    handle = open(snpDataBaseFile)
    
    lineCnt = 0
    while True:
     
        lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
        
        if not lines:
            break  # no more lines to read
        
        for line in lines:
        
            v = line.strip()
            v = v.split("\t")
            
            #some info-lines in Cosmic-files start with #
            if v[0][0] == '#':
                continue
        
            if lineCnt == 0:
                print("SNP data file 1st line:\n ", line)
                print v
                print v[0],v[1],v[3], v[4]
                
                
            #one hot the ref:
            if v[3] == 'A':            
                ref = dataGen.codeA_asArray
            elif v[3] == 'T':     
                ref = dataGen.codeT_asArray
            elif v[3] == 'C':    
                ref = dataGen.codeC_asArray
            elif v[3] == 'G':    
                ref = dataGen.codeG_asArray
            else:
                ref = dataGen.codeW_asArray #wild-card array
                
            #one hot the alt:
            if v[4] == 'A':            
                alt = dataGen.codeA_asArray
            elif v[4] == 'T':     
                alt = dataGen.codeT_asArray
            elif v[4] == 'C':    
                alt = dataGen.codeC_asArray
            elif v[4] == 'G':    
                alt = dataGen.codeG_asArray
            else:
                alt = dataGen.codeW_asArray #wild-card array
            
            infoList.append([v[0], int(v[1]), ref, alt, v[3], v[4]])
                
            lineCnt += 1
            
            
    return np.asarray(infoList)
            



def fetchProbsAtSnps(rootOutput, chrNr, snpInfoArray, snpIndexOffset, predictionArray, qualArray, labelArray, sampledPositions, checkRefBase_b = 1, startAtPosition = 0, endAtPosition = 1e26 , modelFileName = '', genomeIdName = '', snpIdName = '',  writeOut_b = 0, title = '', randomizeAlt_b = 0):
    '''
    startAtPosition, endAtPosition: must be id to those used in the call to encodeGenome (in the dnaNet_dataGen 
    module) from which the encodedGenomeData were had on which, in turn, the predictions (predictionArray) were 
    made. The predictionArray can be the result of assembling predictions made in segments across the genome data
    (encodedGenomeData), by means of the functions predictAccrossGenomeInSegments followed by 
    assemblePredictArrayFromSegments (both in the dnaNet_stats module). 
    
    OBS: the predictions must cover ALL positions from startAtPosition (included) to endAtPosition (not included). The
    code however checks this; if the check shows that the prediction array contains holes, the look-up of the 
    snp-positions will switch to a much slower method.
    
    OBS OBS: see the dnaNet_stats module for a fct that generates an array of pairs of model prob's on a randomly
    picked subset of positions
    '''
    
    if not os.path.exists(rootOutput):

        os.makedirs(rootOutput)
        print "Directory " + rootOutput + " created. Output will be placed there."
        
    
    probsAtSnpList = []
    
    if writeOut_b == 1:
    
        #Write prob's to file:
        writeTo = rootOutput + 'Chr' + str(chrNr) + '_SNPs_Probs.csv'
    
        with open(writeTo, mode='w') as probsFile:
            probsWriter = csv.writer(probsFile, delimiter='\t')
            #Header line with field names: 
            probsWriter.writerow(['#CHR', 'POS', 'REF', 'ALT', 'CONTEXT', 'FILTER', 'A', 'C', 'G', 'T'])    
            
    #Check that the samplePositions arrays are "without holes" (done above for model1):
    L = sampledPositions.shape[0]
    checkDiff = sampledPositions[1:] - sampledPositions[:(L-1)] #should be array of 1's
    check =  np.where(checkDiff > 1)
    Lp = predictionArray.shape[0]
    Lq = qualArray.shape[0]
    slowMethod_b = 0
    checkPassedAs = 0
    if len(check[0]) > 1:
        print "Obs! sampledPositions has %d holes!" % len(check[0])
        print "Will be running slow method!"
        checkPassedAs = 2
        slowMethod_b = 1
    elif Lp != L or Lq != L: #these identities should be ok by construction so superfluous
        print "Obs! sampledPositions and predictionsArray or qualArray have diff shapes"
        print "Will be running slow method!"
        checkPassedAs = 1
        slowMethod_b = 1
    else:
        print "sampledPositions has no holes and arrays have equal lengths"
    
    cntNotQual = 0
    cntDiffRefBase = 0
    cntSnpBaseOffThoQual = 0
    if randomizeAlt_b == 1:
        basesEncodedList = [dataGen.codeA_asArray, dataGen.codeC_asArray, dataGen.codeG_asArray, dataGen.codeT_asArray]

    for i in range(snpInfoArray.shape[0]):

        if snpInfoArray[i][0] != str(chrNr):            
            continue
        
        idx = snpInfoArray[i][1] - snpIndexOffset #subtract snpIndexOffset: snp-info counts first genomic position as snpIndexOffset
    
        
        if idx < startAtPosition or idx >= endAtPosition:
            continue
        
        
        refStr = snpInfoArray[i][4]
        
        altStr = snpInfoArray[i][5]
                
        samplePos = idx - startAtPosition  #the sampled positions are relative to the offset used when reading in the genomic seq
        
        if sampledPositions[L-1] < samplePos: #if the index (~samplePos) goes beyond the sampledPositions: break loop
            lastCoveredSnpIdx = snpInfoArray[i-1][1]
            firstNotCoveredSnpIdx = idx + snpIndexOffset
            print "Break at snp with idx: %d; last covered snp then has index: %d " % (firstNotCoveredSnpIdx , lastCoveredSnpIdx)
            break
        
        if slowMethod_b == 1:
            #find the index of the position in the sampled positions/prediction array (if there):
            samplePosIdx = np.where(sampledPositions == samplePos)
            
            #continue if the snp-position was not sampled
            if len(samplePosIdx[0]) == 0:
                continue
            
            samplePosIdx = samplePosIdx[0][0]
            predProbs = predictionArray[samplePosIdx]
    #            print sampleIdx, predProbs
            qual = qualArray[samplePosIdx]
        else:
            
            samplePosIdx =  samplePos - sampledPositions[0]            
            predProbs = predictionArray[samplePosIdx]
    #            print sampleIdx, predProbs
            qual = qualArray[samplePosIdx]
        
        ref = snpInfoArray[i][2]
        
        alt = snpInfoArray[i][3]
                
        flt = 'PASS'
        probRef = np.dot(ref, predProbs)
        probAlt = np.dot(alt, predProbs)

            
        if probAlt > probRef:
            flt = 'no'
        
        if qual == 0:
            cntNotQual +=1
            continue
    
        if np.max(ref) > 1 or np.max(alt) > 1:
            cntSnpBaseOffThoQual += 1
            continue
        
        #if desired check if the ref-base according to the SNP-data is the same as the base in the sampledGenomeSequence at
        #the current position:
        if checkRefBase_b == 1:
            
            #We could just do it like this ...
            #if refStr != sampledGenomeSequence[samplePos].upper():
            #    cntDiffRefBase += 1
            #... but it is safer to use the labelArray since it matches the predictionArray:
            if not(np.array_equal(ref , labelArray[samplePosIdx])):
                cntDiffRefBase += 1
                
        #for randomizing the alt allele acc to the moel probs.        
        if randomizeAlt_b == 1:
            
            #find index of ref:
            refIdx = np.argmax(ref)
            #pick randomly among {0,1,2,3}\refIdx
            chooseAmong = [0,1,2,3]
            chooseAmong.pop(refIdx)
            takeThisIdx =  np.random.choice(chooseAmong, p = predProbs.take(chooseAmong)/np.sum(predProbs.take(chooseAmong))) #divide: to obtain prob distr -- really just the distr conditional on ref
            alt = basesEncodedList[takeThisIdx]
            probAlt = np.dot(alt, predProbs)
        
        if writeOut_b == 1:
            #Write to file
            #Fields: #CHR	POS	REF	ALT	CONTEXT	FILTER	A	C   G    T
            probsWriter.writerow([str(chrNr), str(idx), refStr, altStr, 'NN', flt, str(predProbs[0]), str(predProbs[1]), str(predProbs[2]), str(predProbs[3])])    
        
        probsAtSnpList.append([samplePosIdx, probRef, probAlt]) #samplePosIdx isthe index at which the prob's where taken
    
    if writeOut_b == 1:
        probsFile.close()
    
    probsAtSnp = np.asarray(probsAtSnpList)
    print "prediction Array shape: ", predictionArray.shape
    print "qual Array shape: ", qualArray.shape
    print "snpInfoArray shape: ", snpInfoArray.shape
    print "probsAtSnp array shape: ", probsAtSnp.shape
    print "Nr of not-qualified positions: ", cntNotQual
    print "Nr of diff ref bases, snp-data vs sampled: ", cntDiffRefBase
    print "Nr of ref or bases in snp-data not-ACGT but qual positions ", cntSnpBaseOffThoQual
    
    dumpFile = rootOutput + 'diagnotstics_snpAnalysis_fetchProbsAtSnps ' + title + ', ' + genomeIdName +  ', ' + snpIdName
    diagDict = {'checkPassedAs (see code for meaning)':checkPassedAs, 'prediction Array shape':predictionArray.shape, 'qual Array shape':qualArray.shape, 'snpInfoArray shape':snpInfoArray.shape, 'probsAtSnp array shape':probsAtSnp.shape, 'index last covered snp index, first not-covered':(lastCoveredSnpIdx, firstNotCoveredSnpIdx), 'Nr of not-qualified positions':cntNotQual, 'Nr of diff ref bases, snp-data vs sampled (for labelArray etc)':cntDiffRefBase }
    pickle.dump(diagDict, open(dumpFile, 'wb'))
        
    return probsAtSnp
        
 

def probsAtSnpAnnos(probsAtSnp, sampledPositions, startAtPosition, annotationDict, repeatComplement_b = 0, repeatAnnoTypes = []):
    '''
    Purpose: derive probs-at-snps arrays for the provided annotations (and always the 'all'
    annotation) based on the input probsAtSnp.
    
    input:
    probsAtSnp: output from fetchProbsAtSNP (array of probs, each entry being (idx, prob-ref, prob-alt))
    sampledPositions: sampled positions from adjacent segments, typically all (as part of 
    output by the dnaNet_stats assemble-predictions fct)
    startAtPosition: the startAtPosition used when calling the predictOnChromosomes fct (from
    which the probs were obtained).
    annotationDict: mapping annotation type to 0/1 indicator array of the annotation 

    
    Returns:
    dict mapping each covered annotation the corr array of probs, each entry being (idx, prob-ref, prob-alt)
    
    
    '''
    #Get annotaion array corr to sampled positions. These must be rel to full chromo seq
    sampledPositionsAbs = sampledPositions + startAtPosition
    sampledPositionsAbs = sampledPositionsAbs.astype(np.int64, casting='unsafe', copy=True)
    annoArrayDict = {}
    for annoType in annotationDict:
        
        annoArrayDict[annoType] = annotationDict[annoType].take(sampledPositionsAbs)
        
        print "Shape of anno-array for anno ", annoType, annoArrayDict[annoType].shape

    #If desired switch to opposite booleans for the given repeatAnnoTypes
    if repeatComplement_b == 1:
        for annoType in repeatAnnoTypes:
            annoArrayDict[annoType] = 1 - annoArrayDict[annoType]


    resultsDict = {}
    resultsDict['all'] = probsAtSnp
    #for the annos:
    for annoType in annotationDict:
        
        resultsDict[annoType] = []
        
    for triple in probsAtSnp:
        
        idx, refProb, altProb = triple 
        
        idx = int(idx)
    
        for annoType in annotationDict:

            if annoArrayDict[annoType][idx] == 1: 
                resultsDict[annoType].append(triple) 
    
    #tr to arrays:        
    for annoType in annoArrayDict:
        
        resultsDict[annoType] = np.asarray(resultsDict[annoType])
    
    return resultsDict   


def snpHisto2D(rootOutput, modelFileName, genomeIdName, snpIdName, probsAtSnp, title = '', saveAtDpi = 100):
    '''Plots as 2d-density the probability of ref base (x) vs probability of alternative allele (y) 
    according to the given model for the snp-data indicated.'''

    #Obs: To fetch out the pairs of probs from the probsAtSnp (ie get rid of the 
    #accompanying index) is simple: the index is first entry, the two probs the next,
    #so we just need to take out the 2nd and 3rd entry of the transposed probsAtSnp:
    #2d histo
    plt.figure()
    plt.title(title + ', ' + genomeIdName +  ', ' + snpIdName)
    hist2dCounts = plt.hist2d(probsAtSnp[:].T[1], probsAtSnp[:].T[2], bins = 50, norm=LogNorm()) 
    plt.xlabel('Prob. of reference base')
    plt.ylabel('Prob. of alternative base')
    plt.colorbar()
    plt.show()
    plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_' + snpIdName + '_hist2dPlot.pdf', dpi = saveAtDpi ) 
    plt.close() 


def snpProbDiffDensity(rootOutput, modelFileName, genomeIdName, snpIdNameList, probsAtSnpList, log_b = 1, title = '', yMax = 1.45, saveAtDpi = 100, colorCntList = '', linestyleList = '', returnSnpData_b = 0, makeDiffDensity_b = 0, makeScatter_b = 0):
    '''Plots density of probability of ref base - probability of alternative allele (or diff of log10 of these prob's) 
    for a list of snp-data and the model's corresponding probabilities for the two alleles.'''
    
    #build a dict from the input data mapping each snpIdName to the data to be plotted:
    snpDataDict = {} 
    cnt = 0
    for probsAtSnp in probsAtSnpList:
        
        #Obs: To fetch out the pairs of probs from the probsAtSnp (ie get rid of the 
        #accompanying index) is simple: the index is first entry, the two probs the next,
        #so we just need to take out the 2nd and 3rd entry of the transposed probsAtSnp:
        
        snpIdName = snpIdNameList[cnt]
        
        try:
            if log_b == 1:
                snpDataDict[snpIdName] =  np.log10(probsAtSnp[:].T[1]) - np.log10(probsAtSnp[:].T[2])
            else: 
                snpDataDict[snpIdName] =  probsAtSnp.T[1] - probsAtSnp.T[2]
        except IndexError:
            print snpIdName
            return
            
        cnt +=1
        
    if colorCntList == '':
        
        colorCntList = range(cnt)
        
    if linestyleList == '':
        
        linestyleList = []
        for i in range(cnt):
            linestyleList.append('solid')

    colors = plt.cm.get_cmap('Set2') #List = plt.cm.hsv(np.linspace(0,1,cnt))
    cnt = 0
    print colors(0)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #Make density plot
    for snpName in snpIdNameList:
        sns.distplot(snpDataDict[snpName], hist = False, kde = True, color = colors(colorCntList[cnt]),
                     kde_kws = {'linewidth': 1.5, 'linestyle':linestyleList[cnt]}, label = snpName)
        cnt +=1 
    # Plot formatting
    ax.axvline(linestyle = '--', color = 'black', ymax = yMax, linewidth=0.5 )
    plt.legend( loc='upper left', fontsize = 'x-small')
    #plt.title(title + ', ' + genomeIdName)
    plt.ylabel('Density',  fontsize = 'small')
    if log_b == 1:
        plt.xlabel('log10(probability of ref) - log10(probability of alt)', fontsize = 'small')
        plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_snpLogPropDiffDensityPlot.pdf', dpi = saveAtDpi) 

    else:
        plt.xlabel('probability of ref - probability of alt',  fontsize = 'small')
        plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_snpPropDiffDensityPlot.pdf' , dpi = saveAtDpi) 

    plt.close() 
    
    #diff between background and case plot:
    if makeDiffDensity_b == 1:
        
        cnt = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #Make density plot
        for i in range(len(snpIdNameList)):
            
            if i%2 == 0:

                X = snpDataDict[snpIdNameList[i]] - snpDataDict[snpIdNameList[i+1]]
                sns.distplot(X, hist = False, kde = True, color = colors(colorCntList[i]),
                             kde_kws = {'linewidth': 1.5, 'linestyle':linestyleList[i]}, label = snpIdNameList[i] + ' minus '  + snpIdNameList[i+1])
                             
                print snpIdNameList[i] + ' minus '  + snpIdNameList[i+1] + "mean %f, std: %f  " % (np.mean(X), np.std(X))  

        # Plot formatting
        ax.axvline(linestyle = '--', color = 'black', ymax = yMax, linewidth=0.5 )
        plt.legend( loc='upper left', fontsize = 'x-small')
        #plt.title(title + ', ' + genomeIdName)
        plt.ylabel('Density',  fontsize = 'small')

        plt.xlabel('variant(prob. of ref - prob. of alt) - background(prob. of ref - prob. of alt)',  fontsize = 'small')
        plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_snpDiffPropDiffDensityPlot.pdf' , dpi = saveAtDpi) 

        plt.close()
        
    #scatter plot
    if makeScatter_b == 1:
        cnt = 0
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #Make scatter plot
        for i in range(len(snpIdNameList)):
            
            if i%2 == 0:
                
                plt.scatter(snpDataDict[snpIdNameList[i]], snpDataDict[snpIdNameList[i+1]], colors(colorCntList[i]), label = snpIdNameList[i] + ' vs ' +  snpIdNameList[i+1])
    
        plt.legend( loc='upper left', fontsize = 'x-small')
        plt.savefig(rootOutput + modelFileName + '_' + genomeIdName + '_snpPropDiffScatterPlot.pdf' , dpi = saveAtDpi) 
            
        plt.close()
    
    
    if returnSnpData_b == 1:    
        return snpDataDict


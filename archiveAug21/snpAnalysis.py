#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:11:30 2019

@author: Christian Grønbæk
"""


'''
Usage:
    
import snpAnalysis as snp


#Read in data:
rootSnp = r'/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/'

chrNr = 22
fileName = 'ALL.chr' + str(chrNr) + '.SNP_27022019.GRCh38.phased.vcf'


snpData = snp.readSNPdata(rootSnp + fileName)





'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import numpy as np

import dnaNet_dataGen as dataGen

import csv


def readSNPdata(snpDataBaseFile):
    
    '''Read in a file containing: chromo-id, postioin in chromo, ref base, alt base.
    
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
            
            infoList.append([int(v[0]), int(v[1]), ref, alt, v[3], v[4]])
                
            lineCnt += 1
            
            
    return np.asarray(infoList)
            



def fetchProbsAtSnps(snpInfoArray, predictionArray, qualArray, chrNr, modelFileName = '', genomeIdName = '', snpIdName = '', title = '', rootOut = ''):
    
    probsAtSnpList = []
    
    #Write prob's to file:
    writeTo = rootOut + 'Chr' + str(chrNr) + '_SNPs_Probs.csv'

    with open(writeTo, mode='w') as probsFile:
        probsWriter = csv.writer(probsFile, delimiter='\t')
        #Header line with field names:
        probsWriter.writerow(['#CHR', 'POS', 'REF', 'ALT', 'CONTEXT', 'FILTER', 'A', 'C', 'G', 'T'])    
            
    
        for i in range(snpInfoArray.shape[0]):
            
            idx = snpInfoArray[i][1]
            
            refStr = snpInfoArray[i][4]
            
            altStr = snpInfoArray[i][5]
            
            predProbs = predictionArray[idx]
            qual = qualArray[idx]
            
            
            ref = snpInfoArray[i][2]
            
            alt = snpInfoArray[i][3]
            
            flt = 'PASS'
            probRef = np.dot(ref, predProbs)
            probAlt = np.dot(alt, predProbs)
            if probAlt > probRef:
                flt = 'no'
            
            if qual == 0:
                continue
            
            #Write to file
            #Fields: #CHR	POS	REF	ALT	CONTEXT	FILTER	A	C   G    T
            probsWriter.writerow([str(chrNr), str(idx), refStr, altStr, 'NN', flt, str(predProbs[0]), str(predProbs[1]), str(predProbs[2]), str(predProbs[3])])    
            
            probsAtSnpList.append([probRef, probAlt])
        
        probsFile.close()
    
    probsAtSnp = np.asarray(probsAtSnpList)
    print "prediction Array shape: ", predictionArray.shape
    print "qual Array shape: ", qualArray.shape
    print "snpInfoArray shape: ", snpInfoArray.shape
    print "probsAtSnp array shape: ", probsAtSnp.shape
    #plot it:
    plt.figure()
    plt.title(title + ', ' + genomeIdName +  ', ' + snpIdName)
    hist2dCounts = plt.hist2d(probsAtSnp[:].T[0], probsAtSnp[:].T[1], bins = 50, norm=LogNorm()) 
    plt.colorbar()
    plt.show()
    plt.savefig(modelFileName + '_' + genomeIdName + '_' + snpIdName + '_hist2dPlot.pdf' ) 
     
    return probsAtSnp
        
        
        
        



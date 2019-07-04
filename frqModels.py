# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian
"""

'''

import frqModels as frqM

#On binf
root = '/isdata/kroghgrp/tkj375/various_python/theano/DNA_proj/results_frqModels/EColi/'


#On pc
root = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/Ecoli/"
root = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/CElegans/inclRepeats/"



file = "frqModel_k4.file"
fileName = root + file

resDict = frqM.readResults(fileName)

comPred = frqM.compPred(resDict)


################################################
## Splitting genome into chromo seq's
################################################


#Human
#On binf
root = '/isdata/kroghgrp/krogh/scratch/db/hg19/'

rootOut = '/isdata/kroghgrp/tkj375/data/DNA/human/hg19/'

file = "hg19.fa"
genomeFile = root + file

outFile = rootOut + 'hg19'

frqM.splitGenomeInChromos(genomeFile, outFile)


#For other species/genome files: change the code to match the content/format of the file

'''

import matplotlib.pyplot as plt

#Split a genome into chromosome seq's:
def splitGenomeInChromos(genomeFile, outFile):
    
    cntChr = 0
    of = open(outFile + 'dummy.fa', 'w') #dummy init
    for line in open(genomeFile):
        
        #Change this depending on genome file (here for hg19 file):
        if line[:4] == '>chr' and (len(line) < 7 or (line[5] !='_' and line[6] !='_')): 
            print line
            
            of.close()
            of = open(outFile + '_' + line[1:].strip() + '.fa', 'w')
            cntChr += 1
            
        of.write(line)
        
    
    print "Found %d chromos" % cntChr

def readResults(fileName):
    
    outDict = {}
    
    for line in open(fileName):
        
        v = line.strip()
        v = v.split(' ')
        
        outDict[v[7]] = map(float, v[:7])
        
    return outDict
    
def compPred(resultsDict):

    avgPred = 0.0
    cnt = 0
    predList = []
    cntList = []
    posList = []
    
    for combi in resultsDict.keys():
        
        cnt += resultsDict[combi][0]
        avgPred += resultsDict[combi][0]*resultsDict[combi][1]
        posList.append(resultsDict[combi][0]*resultsDict[combi][1])
        predList.append(resultsDict[combi][1])
        cntList.append(resultsDict[combi][0])
        
    avgPred = avgPred/cnt
    
    print "Tot cnt: %d " % cnt
    
    print "Avg pred: %f" %avgPred

    plt.figure()
    plt.title("Pred vs positives")
    plt.scatter(predList, posList)
    
    plt.figure()
    plt.title("Sorted counts")
    cntList.sort()
    plt.plot(cntList)

    plt.figure()
    plt.title("Sorted pred")
    predList.sort()
    plt.plot(predList)


    
    
    
        
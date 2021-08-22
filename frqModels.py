# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:17 2017

@author: Christian Grønbæk

"""

'''
The code below -- all in a 'research state' -- was used for part concerning the "k central models" (benchmark models)
of the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

The module is for handling results from the k central models had by A.Krogh's programs. These are described in
the paper

Y.Liang, C.Grønbæk, P.Fariselli, A.Krogh, "Context dependency of nucleotide probabilities and 
variants in human DNA", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.22.453351.
        
Please cite these papers appropriately if you use the code or models -- or parts of it -- in your own work. 


Notes:
        -- If you want to use these models please contact one of us. The code below uses the 
            output of these models as input; to get that output takes running other code not included
            here.
        -- The interpolated versions of the models are coded separately here (below).

Regarding file names:
-- frq_k*.txt with k = 3,4 are just the "average" k central models as had by AK's program.
-- frq_k5_raw.txt is the "average" k=5 central model as had by AK's program.
-- frq_k5.txt is the "interpolated" k=5 central model obtained from frq_k5_raw.txt via interpolation
using frq_k4.txt.

##################################################################################################
# Usage:
##################################################################################################

The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.


import frqModels as frqM

import numpy as np

#On binf
#root = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/onOldCodeVersion/EColi/'
#root = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/onOldCodeVersion/human/hg19/firstRun/'

root = '/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_frqModels/human/'

#On pc
#root = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/Ecoli/"
#root = r"D:/Bioinformatics/various_python/theano/DNA_proj/results_frqModels/CElegans/inclRepeats/"


#Try it out:

file = "frqModel_k4.txt"
fileName = root + file
#read in:
resDict = frqM.readResults(fileName)
#check the obtained accuracy:
comPred = frqM.compPred(resDict)


#Loglikelihood ratio tests and more:

#k 4 vs 3:
file3 = "frqModel_k3.txt"
fileName3 = root + file3
resDict3 = frqM.readResults(fileName3)

file4 = "frqModel_k4.txt"
fileName4 = root + file4
resDict4 = frqM.readResults(fileName4)

dfDiff = 3*np.power(4,8) - 3*np.power(4,6)
pVal, testStat = frqM.logLikelihoodRatioTest(resDict3, resDict4, dfDiff)


#For k= 5 first build the interpolated model from k=4 and the "raw" k=5 model:
file5 = "frqModel_k5_raw.txt"
fileName5 = root + file5
resDict5 = frqM.readResults(fileName5)

#Build interpolated and save it to file:
resultsDictBaseModel = resDict4
resultsDictModel = resDict5
file5interpolated = "frqModel_k5.txt"
fileName5interpolated = root + file5interpolated
outputFilename = fileName5interpolated
alpha = 100
resDict5interpolated = frqM.interpolatedFrqModel(resultsDictBaseModel = resultsDictBaseModel , flankSizeBaseModel = 4, resultsDictModel = resultsDictModel, flankSize = 5,  outputFilename = outputFilename, alpha = alpha)


file5 = "frqModel_k5.txt"
fileName5 = root + file5
resDict5 = frqM.readResults(fileName5)

#k 5 vs 3:
file5 = "frqModel_k5.txt"
fileName5 = root + file5
resDict5 = frqM.readResults(fileName5)

dfDiff = 3*np.power(4,10) - 3*np.power(4,6)
pVal, testStat = frqM.logLikelihoodRatioTest(resDict3, resDict5, dfDiff)

#k4 vs 5:
file5 = "frqModel_k5.txt"
fileName5 = root + file5
resDict5 = frqM.readResults(fileName5)

dfDiff = 3*np.power(4,10) - 3*np.power(4,8)
pVal, testStat = frqM.logLikelihoodRatioTest(resDict4, resDict5, dfDiff)

#k5 raw vs k5 interpolated (may not be considered nested models; try possibly with the non-nested models test)
dfDiff = 0
pVal, testStat = frqM.logLikelihoodRatioTest(resDict5, resDict5interpolated, dfDiff)

################################################
## Splitting genome into chromo seq's (use fct in dnaNet_dataGen instead)
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


import numpy as np 

import scipy.stats as stats

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''AK's encoding:
 const char *lettervector[6] = {
  "SOMETHING IS WRONG",
  "1 0 0 0", // A
  "0 1 0 0", // C
  "0 0 1 0", // G
  "0 0 0 1", // T
  "0 0 0 0"  // N
};
'''

codeA = [1,0,0,0]
codeC = [0,1,0,0]
codeG = [0,0,1,0]
codeT = [0,0,0,1]

codeA_asArray = np.asarray(codeA, dtype ='int8')
codeC_asArray = np.asarray(codeC, dtype ='int8')
codeG_asArray = np.asarray(codeG, dtype ='int8')
codeT_asArray = np.asarray(codeT, dtype ='int8')

#codeA_asArray = np.asarray(codeA, dtype ='float32')
#codeC_asArray = np.asarray(codeC, dtype ='float32')
#codeG_asArray = np.asarray(codeG, dtype ='float32')
#codeT_asArray = np.asarray(codeT, dtype ='float32')


codeA_asInt = 0
codeC_asInt = 1
codeG_asInt = 2
codeT_asInt = 3


#for base pair encoding
codeAT = [1,0]
codeCG = [0,1]

codeAT_asArray = np.asarray(codeAT, dtype ='int8')
codeCG_asArray = np.asarray(codeCG, dtype ='int8')

#for base type encoding (purine/pyrimidine)
codeAG = [1,0]
codeCT = [0,1]

codeAG_asArray = np.asarray(codeAG, dtype ='int8')
codeCT_asArray = np.asarray(codeCT, dtype ='int8')

#for exon/repeat/other coding:
exonicInd =  np.asarray([1, 0,0])
repeatInd =  np.asarray([0,1,0])
otherInd  = np.asarray([0,0,1])

#for repeat/non-repeat coding:
repeatBinInd = np.asarray([1,0])
notRepeatBinInd = np.asarray([0,1])


alphabet = ['A', 'C', 'G', 'T']
alphabetOnehot = np.asarray([codeA_asArray, codeC_asArray, codeG_asArray,codeT_asArray])


#Split a genome into chromosome seq's. But use the fct in dnaNet_dataGen instead:
def splitGenomeInChromos(genomeFile, outFile):
    
    cntChr = 0
    of = open(outFile + 'dummy.fa', 'w') #dummy init
    for line in open(genomeFile):
        
        #Change this depending on genome file (here for hg19 file):
        if line[:4] == '>chr' and (len(line) < 7 or (line[5] !='_' and line[6] !='_')): 
            print(line)
            
            of.close()
            of = open(outFile + '_' + line[1:].strip() + '.fa', 'w')
            cntChr += 1
            
        of.write(line)
        
    
    print("Found %d chromos", cntChr)

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
    
    for context in resultsDict.keys():
        
        cnt += resultsDict[context][0]
        avgPred += resultsDict[context][0]*resultsDict[context][1]
        posList.append(resultsDict[context][0]*resultsDict[context][1])
        predList.append(resultsDict[context][1])
        cntList.append(resultsDict[context][0])
        
    avgPred = avgPred/cnt
    
    print("Tot cnt: %d ", cnt)
    
    print("Avg pred: %f" ,avgPred)

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



def logLikelihood(resultsDict):
    '''Computes the log-likelihood of the results from a k-mer model.
    
    Input:
        resultsDict: as output by readResults for the given k-mer model.
        
    Obs: if in a context there are no occurrences of a particular base (B) 
    we have p(B)= 0; log(p(B)) is then not defined but p(B)*log(p(B)) is (can 
    be set to) zero (as x*logx converges to 0 in the 0-limit).'''
    
    LL = 0
    cntTot = 0
    for context in resultsDict.keys():
        
        cnt, predProb, tech, pA, pC, pG, pT = resultsDict[context]
        
        
        if pA > 0:
            LL += cnt*pA*np.log(pA)
        if pC > 0:
            LL += cnt*pC*np.log(pC)
        if pG > 0:
            LL += cnt*pG*np.log(pG)
        if pT > 0:
            LL += cnt*pT*np.log(pT)
        
        cntTot += cnt
    
    print("Total number of positions ", cntTot)
    
    return LL



def chiSquared(x, df):

    return stats.chi2.cdf(x,df) 


def logLikelihoodRatioTest(resultsDict1, resultsDict2, dfDiff):
    
    '''resultsDict1: from more simple model of the two.'''

    ll1 = logLikelihood(resultsDict1)
    ll2 = logLikelihood(resultsDict2)  
    
    testStat = -2*ll1 + 2*ll2
    
    print("Test stat value is: %d ", testStat)
    
    pVal = 1.0 - chiSquared(testStat, dfDiff)
    
    print("p val is: %lf" , pVal)
    
    return pVal, testStat
        
     

        
    
    
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
    resDict = readResults(fileName)
    
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
            #prob, and zeros on the remaining three:
            maxAt = np.argmax(np.asarray(value))
            newValue = alphabetOnehot[maxAt]
            
            outDict[newWord] = newValue
        

    return outDict
    
    
def interpolatedFrqModel(resultsDictBaseModel, flankSizeBaseModel, resultsDictModel, flankSize,  outputFilename = '', alpha = 100):
    '''To use for generating the "final" k-mer  models for k above 4, by interpolating
    from the (k-1)-mer model (eg the 5-mer model is obtained by interpolation from the 4-mer
    model).

    resultsDictModel: dict obtained by readResults for the model we wants to improve by 
    interpolating with the one-k-lower model
    
    resultsDictBaseModel: dict obtained by readResults for the one-k-lower model

    output: a dict of the same structure as from readResults; the dict gives the 
    data of the interpolated model.
    Further, the interpolated model is written to a file (outputFilename) in the
    same format/structure as the files for the input k-mer models have.
    
    '''    
    
    outDict = {}
    
    #open output file in write mode for flushing it
    outFile = open(outputFilename, "w")
    outFile.close() #file flushed
    #reopen in append mode:
    outFile = open(outputFilename, "a")
    
    
    for word in resultsDictModel:
       
       #look-up results from base model and model to adjust:
       wordBase = word[(flankSize - flankSizeBaseModel):(flankSize + flankSizeBaseModel +1)]
       cnt_base, maxProb_base, ent_base, pA_base, pC_base, pG_base, pT_base  = resultsDictBaseModel[wordBase]
       
       cnt, maxProb, ent, pA, pC, pG, pT  = resultsDictModel[word]
       
       cntA = pA*cnt  
       pA_new = float(cntA + alpha*pA_base)/(cnt + alpha)
       
       cntC = pC*cnt    
       pC_new = float(cntC + alpha*pC_base)/(cnt + alpha)
       
       cntG = pG*cnt  
       pG_new = float(cntG + alpha*pG_base)/(cnt + alpha)
       
       cntT = pT*cnt  
       pT_new = float(cntT + alpha*pT_base)/(cnt + alpha)
       
#       outDict[word] = pA_new, pC_new, pG_new, pT_new 
       outDict[word] = cnt, max([pA_new, pC_new,pG_new,pT_new]), ent, pA_new, pC_new, pG_new, pT_new #ent is not updated!

       #write result to file: 
       outString =  str(int(cnt)) + ' ' + str(max([pA_new, pC_new,pG_new,pT_new])) + ' ' + str(ent) + ' ' + str(pA_new) + ' ' + str(pC_new) + ' ' + str(pG_new) + ' ' + str(pT_new) + ' ' + word + '\n'
       outFile.write(outString)         
       
    return outDict
       
        
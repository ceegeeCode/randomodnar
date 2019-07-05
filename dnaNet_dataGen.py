#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:01:36 2019

@author: Christian Grønbæk
"""

'''
Usage:
    
The main functions right now are

readGenome
encodeGenome
genSamplesForDynamicSampling_I

and (for language models)
genSamplesDirectlyFromGenome
 

For input/output see their doc-strings. What they do:    

readGenome: reads in a .fa formatted file. The human genome assembly hg19.fa, for instance. This
be downloaded via http://hgdownload.cse.ucsc.edu/downloads.html#human (it's explained there how, it's
just a wget really).    
    

encodeGenome: reads in putput from readGenome and spits out an encoded versions; encodings: one-hot or
as integers (0,1,2,3). At the very top of the code you'll find how the four letters (A C G T) 
are encoded (A, C, G, T corr's to 0,1,2,3 etc)  
    
    
genSamplesForDynamicSampling_I: is the sampling fct used for the dynamical sampling (e.g in the fct
allInOneWithDynSampling_ConvLSTMmodel found in dnaNet_v7)

genSamplesDirectlyFromGenome: generates samples from a genome string (read in by readGenome)  


'''

import numpy as np
import sys


from random import shuffle



###################################################################
## Defs and Utils
###################################################################

#codeA = np.array([1,0,0,0])
#codeG = np.array([0,1,0,0])
#codeC = np.array([0,0,1,0])
#codeT = np.array([0,0,0,1])


'''
AK's encoding:
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


def oneHotLetter(letter):
    
    '''Returns the one-hot encoding of the given letter following the
    definitions above.'''
                            
    if letter == 'A':            
        return codeA_asArray
    elif letter == 'T':     
        return codeT_asArray
    elif letter == 'C':    
        return codeC_asArray
    elif letter == 'G':    
        return codeG_asArray


def invOneHotLetter(oneHotArray4):
    
    '''Returns the letter corr to the given one-hot encoding following the
    definitions above.'''
                            
    if np.array_equal(oneHotArray4, codeA_asArray):
        return 'A' 
    elif np.array_equal(oneHotArray4, codeT_asArray):
        return 'T'
    elif np.array_equal(oneHotArray4, codeC_asArray):
        return 'C'
    elif np.array_equal(oneHotArray4, codeG_asArray):
        return 'G'


def intLetter(letter):
    
    '''Returns the one-hot encoding of the given letter following the
    definitions above.'''
                            
    if letter == 'A':            
        return codeA_asInt
    elif letter == 'T':     
        return codeT_asInt
    elif letter == 'C':    
        return codeC_asInt
    elif letter == 'G':    
        return codeG_asInt

def invIntLetter(intCode):
   
    '''Returns the letter corr to the given integer encoding following the
    definitions above.'''
    
    if intCode == codeA_asInt:
        return 'A' 
    elif intCode == codeT_asInt:
        return 'T'
    elif intCode == codeC_asInt:
        return 'C'
    elif intCode == codeG_asInt:
        return 'G'
    
    
def invNumpyIntInt(intCode):
   
    '''Returns the integer corr to the given integer encoding following the
    definitions above.'''
    
    if intCode == codeA_asInt:
        return 0 
    elif intCode == codeT_asInt:
        return 1
    elif intCode == codeC_asInt:
        return 2
    elif intCode == codeG_asInt:
        return 3

def complementLetter(x):
    
    if x == 'A':
        return 'T'
    if x == 'T':
        return 'A'
    if x == 'C':
        return 'G'
    if x == 'G':
        return 'C'
        
    if x == 'a':
        return 't'
    if x == 't':
        return 'a'
    if x == 'c':
        return 'g'
    if x == 'g':
        return 'c'

def complement(x):
    
    if np.array_equal(x, codeA_asArray):
        return codeT_asArray
    elif np.array_equal(x, codeT_asArray):
        return codeA_asArray
    elif np.array_equal(x, codeC_asArray):
        return codeG_asArray
    elif np.array_equal(x, codeG_asArray):
        return codeC_asArray


def complementArray(x):
    
    if np.array_equal(x, codeA_asArray):
        return codeT_asArray
    elif np.array_equal(x, codeT_asArray):
        return codeA_asArray
    elif np.array_equal(x, codeC_asArray):
        return codeG_asArray
    elif np.array_equal(x, codeG_asArray):
        return codeC_asArray
        

def complementInt(x):
    
    if np.array_equal(x, codeA_asInt):
        return codeT_asInt
    elif np.array_equal(x, codeT_asInt):
        return codeA_asInt
    elif np.array_equal(x, codeC_asInt):
        return codeG_asInt
    elif np.array_equal(x, codeG_asInt):
        return codeC_asInt
        

def basePair(x): #for Watson-Crick base pair identification
    
    if np.array_equal(x, codeA_asArray):
        return codeAT_asArray
    elif np.array_equal(x, codeT_asArray):
        return codeAT_asArray
    elif np.array_equal(x, codeC_asArray):
        return codeCG_asArray
    elif np.array_equal(x, codeG_asArray):
        return codeCG_asArray


def basePairType(x): #for purine/pyrimidine identification
    
    if np.array_equal(x, codeA_asArray):
        return codeAG_asArray
    elif np.array_equal(x, codeT_asArray):
        return codeCT_asArray
    elif np.array_equal(x, codeC_asArray):
        return codeCT_asArray
    elif np.array_equal(x, codeG_asArray):
        return codeAG_asArray

def complementArrayBasepairType(x):
    
    if np.array_equal(x, codeAG_asArray):
        return codeCT_asArray
    elif np.array_equal(x, codeCT_asArray):
        return codeAG_asArray
    

#For transforming encoding from 1d-arrays to 2d-arrays: the one-hot encoding
#is changed from four letters to the sixteen combinations of two letters     
def seqToPict(inputArray, flankSize = 50):
    
    print("I'm converting format: from seqeunces to pictures") 
    
    nrSamples, seqLength, alphabetSize = inputArray.shape
    
#    nrSamples = 100

    outArray = np.zeros(shape = (nrSamples, flankSize, flankSize, alphabetSize, alphabetSize), dtype = np.int8)
        
    
    for i in range(nrSamples):
        
        for j in range(flankSize):
            
            idx1 = np.argmax(inputArray[i][j]) #the index of 1 in this "A/T/C/G"
            
            for k in range(flankSize):
                
                idx2 = np.argmax(inputArray[i][k + flankSize]) #the index of 1 in this "A/T/C/G"
                
                outArray[i][j][k][idx1][idx2] = 1.0 #put a 1 at this position of the 4-by-4 matrix
#        if i > 10:
#            break

    outArray = np.reshape(outArray, newshape = (nrSamples, flankSize, flankSize, alphabetSize*alphabetSize))

    print("... format conversion done!")

    return outArray
                
def intSeqConvertToOneHotSeq(inputArray, inverse_b = 0):
    '''
    inverse_b: 0: integer arrays to one-hot arrays; 1: the other way
    
    '''
    
    #Convert from integer representation to one-hot:
    if inverse_b == 0:
        
        nrSamples, seqLength = inputArray.shape
        
    
    #    nrSamples = 100
    
        outArray = np.zeros(shape = (nrSamples, seqLength, 4), dtype = np.int8)

        for i in range(nrSamples):
        
            for j in range(seqLength):
                
                if inputArray[i][j] == 0:
                    
                     outArray[i][j] = codeA_asArray
                
                elif inputArray[i][j] == 1:
                    
                     outArray[i][j] = codeT_asArray
                     
                elif inputArray[i][j] == 2:
                    
                     outArray[i][j] = codeC_asArray
                     
                elif inputArray[i][j] == 3:
                    
                     outArray[i][j] = codeG_asArray
                
                
                
        
    #Convert from one-hot representation to integer:
    elif inverse_b == 1:
        
        nrSamples, seqLength, alphabetSize = inputArray.shape
    
    #    nrSamples = 100
    
        outArray = np.zeros(shape = (nrSamples, seqLength), dtype = np.int8)
        
        for i in range(nrSamples):
        
            for j in range(seqLength):
            
                idx = np.argmax(inputArray[i][j]) #the index of 1 in this one-not encoded "A/T/C/G"
                
                outArray[i][j] = idx



    return outArray


def delta4(n):
    
    x = np.zeros(4)
    x[n] = 1
    return x
        
        

    
    

###################################################################
## Reading in data from external source
###################################################################

def getData(fname,letterShape,nout,outputType=float):
    '''
    fname = input file name
    letterShape = dimension of the word encoding, for dna letterShape = 4
    nout =  number of outputs, here 4
    outputType = float by default  
    '''
    x=[]
    y=[]
    k = 0
#    v3 = []
    for line in open(fname):
        
        if k == 10000:
            print("nr of records %d" % k)
            return np.array(x), np.array(y)

        v = map(float,line.split())
#        print line
#        print len(v)
#        print range(len(v[:-nout])/letterShape)
#        for i in range(len(v[:-nout])/letterShape):
#            v3.append(v[i*letterShape:(i+1)*letterShape])
#            if i < 10:
#                print v3
        v2 = [ v[i*letterShape:(i+1)*letterShape] for i in range(len(v[:-nout])/letterShape) ] 
#        v2 = [ v[i:i+letterShape] for i in range(len(v[:-nout])/letterShape) ] 
#        if v2 == v3:
#            print "ok"
        x.append(v2)
#        print x
        y.append(map(outputType,v[-nout:]))
#        print y
        k += 1
    print("nr of records %d" % k)
    return np.array(x), np.array(y)


def shuffleFlanks(v, flankSize, shuffleLength = 1, inner_b = 0):
    ''' Returns shuffled v: shuffles the 2*shuffleLength worth of
    elements either in the middlepart of v (inner_b = 1) or in the 
    outer part of the flanks'''

    vShuffled = v

    if inner_b == 0:
        
        idxs = range(shuffleLength)
        rightPart = 2*flankSize - shuffleLength
        idxs.extend(range(rightPart,2*flankSize)) #2*flankSize = len(v)
        
        x = np.asarray(v).take(idxs, axis = 0)
        np.random.shuffle(x)
        
        vShuffled = v
        vShuffled[:shuffleLength] = x[:shuffleLength]
        vShuffled[rightPart:] = x[shuffleLength:]
        
    else: #shuffle the 2*shuffleLength long central part
    
        idxs = range(flankSize - shuffleLength, flankSize + shuffleLength)
        x = np.asarray(v).take(idxs, axis = 0)
        np.random.shuffle(x)
        vShuffled = v
        vShuffled[flankSize - shuffleLength:flankSize + shuffleLength] = x
        
        
#    return vShuffled
        
        
    


def getData2(fname,
             letterShape,
             nout,
             loadRecsInterval = [0,10000], 
             augmentWithRevComplementary_b = 1, 
             outputType=float,
             convertToPict_b = 0,
             shuffle_b = 0, 
             inner_b = 1, 
             shuffleLength = 5, 
             getFrq_b = 1, 
             rnd_label_b = 0, 
             customFlankSize_b = 0, 
             customFlankSize = 25):
    '''
    fname = input file name
    letterShape = dimension of the letter encoding, for dna letterShape = 4
    nout =  number of outputs, here 4
    outputType = float by default  
    augmentWithRevComplementary_b: for each input list of nucleotide letters (ACGT) generate and add the reversed and complemented list to the output (the label added being the complemented original label)
    shuffle_b: if = 1 the flanking sequences will be randomly shuffled; for testing
    rnd_label_b: if 1 the lables will be drawn at random (from ACGT)
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled
    
    '''
    x=[]
    y=[]
    k = 0
    
    cntA = 0
    cntT = 0
    cntC = 0
    cntG = 0
    
    frqA = 0
    frqT = 0
    frqC = 0
    frqG = 0
    
#    v3 = []
    for line in open(fname):
        if k < loadRecsInterval[0]:
            k += 1
            continue
        if k == loadRecsInterval[1]:
            
            print("nr of records %d", len(x))
            
            if getFrq_b > 0.5:
                
                frqA = cntA/(k - loadRecsInterval[0])
                frqT = cntT/(k - loadRecsInterval[0])
                frqC = cntC/(k - loadRecsInterval[0])
                frqG = cntG/(k - loadRecsInterval[0])
                
                print( "A frq: %f" % frqA)
                print("T frq: %f" % frqT)
                print("C frq: %f" % frqC)
                print("G frq: %f" % frqG)
            
            return np.array(x), np.array(y)

        v = map(float,line.split())
        
        #set flanksize if desired (only do this at first line to read in):
        
        if k == loadRecsInterval[0]:

            flankSize = len(v[:-nout])/(2*letterShape)
            
            if customFlankSize_b == 1:
                leftFlankStart = flankSize - min(flankSize,customFlankSize)
                rightFlankEnd = flankSize + min(flankSize,customFlankSize)
                    
            else:
                leftFlankStart = 0
                rightFlankEnd = 2*flankSize
        
        
#        print line
#        print v
#        print len(v)
#        print range(len(v[:-nout])/letterShape)
#        for i in range(len(v[:-nout])/letterShape):
#            v3.append(v[i*letterShape:(i+1)*letterShape])
#            if i < 10:
#                print v3        
        
        v2 = [ v[i*letterShape:(i+1)*letterShape] for i in range(leftFlankStart, rightFlankEnd) ] 

 
#        v2 = [ v[i:i+letterShape] for i in range(len(v[:-nout])/letterShape) ] 
#        if v2 == v3:
#            print "ok"
        #For testing shuffling is an option; there the flanks are shuffled and a random element in the shuffled flanks are
        #taken as label; else the label is the last entry in the read-in list and everthing before it defines the flanks:
        if shuffle_b > 0.5:
#            print "I shuffle ..."
#            print v2
            if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                shuffleFlanks(v = v2, flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                shuffleFlanks(v = v2, flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            else:
                shuffle(v2)                
                
#            print v2
#            return 0,0
#            yRndIdx = np.random.randint(0,len(v2)) #pick random integer betweem 0 and 2*length of flank, i.e. len(v2)
#            y.append(v2[yRndIdx])
#            print y
#            return 0,0
#        else:
#            y.append(map(outputType,v[-nout:]))
#            print y
            
        x.append(v2)
#        print x
        y.append(map(outputType,v[-nout:]))
#            print y
#        return 0,0
        k += 1

        #Get frequnecies if desired:
        if getFrq_b > 0.5:
#            for j in range(len(v2)):
            if v[-nout:] == codeA:
                cntA +=1. 
            elif v[-nout:] == codeT:
                cntT +=1. 
            elif v[-nout:] == codeC:
                cntC +=1. 
            elif v[-nout:] == codeG:
                cntG +=1.
                
        
        #add reversed and complemented sequence: 
        if augmentWithRevComplementary_b == 1:
            v2Rev = v2[::-1]
            v2RevAug = map(complement, v2Rev)
            x.append(v2RevAug)
            y.append(map(outputType,complement(v[-nout:])))


    
    print("nr of records %d" % len(x))
    
    if getFrq_b > 0.5:
        
        frqA = cntA/(k - loadRecsInterval[0])
        frqT = cntT/(k - loadRecsInterval[0])
        frqC = cntC/(k - loadRecsInterval[0])
        frqG = cntG/(k - loadRecsInterval[0])
                
        print("A frq: %f" % frqA)
        print("T frq: %f" % frqT)
        print("C frq: %f" % frqC)
        print("G frq: %f" % frqG)
        
    X, Y = np.array(x), np.array(y)
        
    if convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        
    return X, Y
    

def genRandomGenome(length, fileName, letters_b = 0, on_binf_b = 0):
    '''Generates a random sequence of the four bases of length the input length.
    Writes to file with name fileName.
    letters_b: if = 1 the output is a string of letters in alphabet ACGT'''
    
    alphabet = 'ACGT'
    
    codeList = [codeA, codeT, codeC, codeG]
    
    lAlphabet = len(alphabet)
    lCodelist = len(codeList)
    
    #flush file by opening it in write mode and closing it again; then open it in append mode
    outFile = open(fileName, 'w')
    outFile.close()
    outFile = open(fileName, 'a')
    
    
    if letters_b == 1:

        line = ''
        
        for i in range(length):
            
            #get a random integer between 0 (incl) and four (excl)
            idx = np.random.randint(0,4)
            line += alphabet[idx]
            
            
            if len(line)%100 == 0:
                            
                outFile.write(line)
                outFile.write('\n')
                #reset:
                line = ''
                
    else:
        
        line = []
        
        randIntArray = np.zeros(shape = length) #for checking the distr of the random int
        
        for i in range(length):
            
            #get a random integer between 0 (incl) and four (excl)
            idx = np.random.randint(0,4)
            randIntArray[i] = idx
            line.append(codeList[idx])
            
#            print len(line)
            
            if len(line) == 100:
                
                for j in range(len(line)):
                    for k in range(lCodelist):  
                        outFile.write(str(line[j][k]))
                outFile.write('\n')
                #reset:
                line = []
                
                
        #plot histogram of the smapled ints:
        if on_binf_b == 0:
            n, bins, patches = plt.hist(randIntArray, bins = 4)
            print( "Frequencies: %lf" % n/length)
    


    
###########################################################
    ### For testing on genome
###########################################################


def readGenome(fileName, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 10, 
               startAtPosition = 0,
               endAtPosition = int(1e26), #some gigantic number
               outputAsDict_b = 0,
               outputGenomeString_b =0,
               randomChromo_b = 0, 
               avoidChromo = []):
    
    '''
    Input:
       fileName: path to genome file (.fa format expected)
       exonicInfoBinaryFileName: file containing exonic information for all positions in the genomic file (fileName)
       chromoNameBound = 10: this integer sets a bound on the name-length of the chromosomes to be read in (In some genome files 
       (e.g hg19.fa) "extra" chromosomes are included, typically having names longer than the "true" chromos.)
       startAtPosition: first position in genome seq to read in
       endAtPosition: last position in genome seq to read in
       outputAsDict_b: if 1 the output is a dictionary mapping each (covered) chromo to the read-in sequence; else (if 0) 
       the output is the read-in sequence as a string 
       randomChromo_b: if 1 the seq of a randomly chosen chromo is read in  
       avoidChromo: this list contains names of chromos to be avoided (e.g. sex chromo's, mitochondrial)
        
    Output:
        
        If outputAsDict_b = 0: a tuple of two strings and two lists: Xall, X, Xrepeat, Xexonic
        
        where
        Xall: genome string in capital letters (ie no repeat masking)
        X: genome string as it is (ie with repeat masking, if exists in input file)
        Xrepeat: list of 0/1 corr to genome string/positions; 1 if position is a repeat, 0 else
        Xexonic: list of exonic-codes corr to genome string/positions; e.g 0 if position is intergenic, 1 if exonic, 2 if intronic   
        
        If outputAsDict_b = 1: a tuple of five dictionaries: XchrAllDict, XchrDict, XchrRepeatDict, XchrExonicDict, XchrNsDict
        
        each mapping a chromo to its sequence (or whatever was loaded in of it) or list of annotations. the first four
        correspond to the strings/listsoputput for outputAsDict_b = 0:
        
        XchrAllDict: see Xall
        XchrDict: see X
        XchrRepeatDict: see Xrepeat
        XchrExonicDict: see Xexonic
        XchrNsDict: for each chromo seq read in (key), a list of two strings: the heading N's and the trailing N's (if any)
        of the seq read in.
        
    '''

    
    XexonicChromo = '' #to contain the full sequence of all 0/1 exonic indicators for (a desired part of) a chromo
    Xall = '' #to contain the full sequence of all letters in the read-in genome
    XexonicAll = '' #to contain the full sequence of all 0/1 exonic indicators in the read-in genome
    X = '' #to only contain the ACGT's of the genome
    Xrepeat = [] #to contain the indicator of a position being in a repeat or not
    Xexonic = [] #to contain the indicator of a position being exonic or not
    Xchr = '' #to only contain the ACGT's in one chromosome of the read-in genome
    XchrExonic = '' #to only contain the exonic-info in one chromosome of the read-in genome
    XchrRepeat = [] #to contain the indicator of a position being in a repeat or not
    XchrExonic = [] #to contain the indicator of a position being exonic or not
    XchrAllDict = {} #dict to map: each chromosome to its sequence
    XchrExonicAllDict = {} #dict to map: each chromosome to its exonic-info sequence
    XchrDict = {} #dict to map: each chromosome to list [its length, sequence(only ACGT's)]
    XchrRepeatDict = {} #repeat info corr to XchrDict
    XchrExonicDict = {} #exonic info corr to XchrDict
    XchrNsDict = {} #to collect the number of heading,trailing ends for each chromo
    lineCnt = 0
    lenXall = 0
    lenX = 0
    lenXchrAll = 0
    lenXchr = 0
    
    print("Reading in genome data ... ")
    print("Only considering data following fasta header lines (: chromo names \n for eucaryots) of length < %d" %  chromoNameBound)



    #First Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length:            
    lenChr = 0
    chromoList = []
    exonicInfoList = []
    currChromo = ''
    for line in open(fileName):
        
        v = line.strip()
        
        #skip line starting with a '>'; these header lines will preceed the dna-seq
        #or the dna-seq for each chromosome 
        if v[0] == '>' and len(v[1:]) < chromoNameBound:
            
            if currChromo != '':
                chromoList.append([currChromo, lenChr])    
            
            currChromo = v[1:]
            print("Found data for this chromosome: %s" % currChromo)
            lenChr = 0
            
            if exonicInfoBinaryFileName != '':
                    
                #get the exonic info and find the content for this chromo:
                exonicInfoFile = open(exonicInfoBinaryFileName, 'r')
                hit_b = 0
                for row in exonicInfoFile:
#                        print row[:20]
                    if row != line:
                        continue
                    else:
                        hit_b = 1
#                            print "Hit!"
                        break
                #The exonic info for the chromo is now at the following line:
                if hit_b == 1:
                    XexonicChromo = exonicInfoFile.next()
                    
                exonicInfoList.append([currChromo, len(XexonicChromo)])
                        
                        
        else:
            lenChr += len(v)

    #record for lastly passed chromo too:
    chromoList.append([currChromo, lenChr])    
            
    print(chromoList)
    print(exonicInfoList)

    
    accumulatedLength = 0
    if outputAsDict_b == 0:
        
        #Loop through the file to first find the total length of the dna:
        for line in open(fileName):
            
            v = line.strip()
            
            if lineCnt == 0:
                print("Genome data file 1st line:\n ", line)
            
            #skip line starting with a '>'; these header lines will preceed the dna-seq
            #or the dna-seq for each chromosome 
            if v[0] == '>' and len(v[1:]) < chromoNameBound:
                
                print("Will consider data for this chromosome: ", line)
                
                posInChr = 0
                
                if exonicInfoBinaryFileName != '':
                    
                    #get the exonic info and find the content for this chromo:
                    exonicInfoFile = open(exonicInfoBinaryFileName, 'r')
                    hit_b = 0
                    for row in exonicInfoFile:
#                        print row[:20]
                        if row != line:
                            continue
                        else:
                            hit_b = 1
#                            print "Hit!"
                            break
                    #The exonic info for the chromo is now at the following line:
                    if hit_b == 1:
                        XexonicChromo = exonicInfoFile.next()
#                        print XexonicChromo[:50]
#                    else:
#                        print "No hit!"
                    
#                    print "XexonicChromo first 1000: ", XexonicChromo[:1000]
                    
                else: #if no exon-info file was provided, we use 0:
                    
                    print("OBS: no file containing exonic info was provided, so exonic status is set to 0 throughout!")
                    for i in range(lenXall, endAtPosition +1):
                        
                        XexonicChromo += str(0)
                    

                    
            else: #if v[0] != '>' ie line is not chromo name (fasta format)
    
    #            print v
    
    #            v = map(float,line.split())
                accumulatedLength += len(v)
                posInChr += len(v)
                if accumulatedLength >= startAtPosition:
                                            
                    
                    if accumulatedLength <= endAtPosition:
                        
                        Xall += v

                        XexonicAll += XexonicChromo[(posInChr - len(v)):posInChr]
                    
        #            print "Xall at %d: %s" % (lineCnt, Xall)
        #            if lineCnt ==2:
        #                return 
                        lenXall += len(v)
                        
                    
                        
                    else:
                        break
                    
            lineCnt += 1 
            
            
            
        print("Length of genome sequence read in:%d" % lenXall)
        print("Length of exonic-info sequence read in:%d" % len(XexonicAll))
        if lenXall != len(XexonicAll):
            raw_input("Warning: lengths of exonic info and dna-seq differ!")
        
        #not all letters are ACGT!:
        for i in range(lenXall):
                            
            try:
                if Xall[i] == 'A':            
                    X += 'A'
                    Xrepeat.append(0)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'T':     
                    X += 'T'
                    Xrepeat.append(0)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'C':    
                    X += 'C'
                    Xrepeat.append(0)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'G':    
                    X += 'G'
                    Xrepeat.append(0)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'a':            
                    X += 'A'
                    Xrepeat.append(1)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 't':     
                    X += 'T'
                    Xrepeat.append(1)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'c':    
                    X += 'C'
                    Xrepeat.append(1)
                    Xexonic.append(XexonicAll[i])
                elif Xall[i] == 'g':    
                    X += 'G'
                    Xrepeat.append(1)
                    Xexonic.append(XexonicAll[i])
                        
            except IndexError:
                print("Letter is: %s so not ACGTacgt at: %d" % (Xall[i],i))
                
                
        #If desired the letters will be "one-hot" encoded:
        lenX = len(X)
        print("Length genome sequence, only ACGT's:%d" % lenX)
        
        
        
        return Xall, X, Xrepeat, Xexonic





    else: #outputAsDict_b != 0:
    
        #If desired we only read in data from one randomly chosen chromosome
        
        if randomChromo_b == 1:
            
            chromoList = []
            
            #First Loop through the file to find the diff chromo's:            
            for line in open(fileName):
                
                v = line.strip()
                
                #skip line starting with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
                if v[0] == '>' and len(v[1:]) < chromoNameBound:
                    currChromo = v[1:]
                    print("Found data for this chromosome: ", currChromo)
                    chromoList.append(currChromo)
                    continue
            
            lChromos = len(chromoList)
            #pick a random one, though redo if the one found is in the avoidChromo's list:
            getDataForThisChromOnly = ''
            while getDataForThisChromOnly == '':
                idx = np.random.randint(lChromos)
                if avoidChromo.count(chromoList[idx]) == 0 and len(chromoList[idx]) < chromoNameBound:
                    getDataForThisChromOnly = chromoList[idx]
                    
            print("Picked randomly this chromo from which the data will be read: %s" % getDataForThisChromOnly )
        
        
        #Loop through the file to record the sequence(s):
        currChromo = ''
        lCurrChromo = 0
        
        #divide in two cases: one reading in data for each chromo and one where only
        #data for the randomly chosen chromo is read in:
        if randomChromo_b != 1:
            
            for line in open(fileName):
                
                v = line.strip()
                
                #lines starting with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
                if v[0] == '>':
                                    
                    #len(currChromo) < chromoNameBound allows avoiding reading in chromo's with long names
                    if v[1:] != currChromo and lCurrChromo < chromoNameBound: 
                        
                        #init/reset
                        accumulatedLength = 0
                        lenXchrAll = 0

                        currChromo = v[1:]
                        lCurrChromo = len(currChromo)
                        XchrAllDict[currChromo] = ''                        
                        print("Will read in data from chromosome: %s " % currChromo)

                        if exonicInfoBinaryFileName != '':
                            
                            #get the exonic info and find the content for this chromo:
                            exonicInfoFile = open(exonicInfoBinaryFileName, 'r')
                            
                            hit_b = 0
                            for row in exonicInfoFile:
    
                                if row != line:
                                    continue
                                else:
    #                                print row[:20]
                                    hit_b = 1
                                    break
                            #The exonic info for the chromo is now at the following line:
                            if hit_b == 1:

                                XexonicChromo = exonicInfoFile.next()
#                                print XexonicChromo[:50]
                        
                        else: #if no exon-info file was provided, we use 0:
                            
                            print("OBS: no file containing exonic info was provided, so exonic status is set to 0 throughout!")

                            for i in range(endAtPosition +1):
                                XexonicChromo += str(0)
                                
                        XchrExonicAllDict[currChromo] = ''
 

                
                elif lCurrChromo < chromoNameBound and accumulatedLength <= endAtPosition:  #if lineCnt > 0: #first line is header (fasta format)
                    
                    accumulatedLength += len(v)
                    
                    if accumulatedLength >= startAtPosition:
                            
                        XchrAllDict[currChromo] += v
                        lenXchrAll += len(v)
                        
                        XchrExonicAllDict[currChromo] += XexonicChromo[(accumulatedLength - len(v)):accumulatedLength]
                        
                if lineCnt%1000000 == 0:
                        print("Have read %d lines" % lineCnt)
                
                lineCnt += 1
                
                
                
        
        elif randomChromo_b == 1:
        
            
            
            #Now get read in and fetch the sequence data etc
            currChromo = ''
            for line in open(fileName):
                
                v = line.strip()
                
                #skip line starting with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
                if v[0] == '>':

                    if v[1:] == getDataForThisChromOnly:

                        currChromo = getDataForThisChromOnly
                        
                        #init/reset
                        accumulatedLength = 0
                        lenXchrAll = 0

                        XchrAllDict[currChromo] = ''
                        print("Will read in data for chromosome: ", currChromo)

            
                        #fetch the exonic info for the picked chromo (= currChromo):
                        if exonicInfoBinaryFileName != '':
                                    
                            #get the exonic info and find the content for this chromo:
                            exonicInfoFile = open(exonicInfoBinaryFileName, 'r')
                            hit_b = 0
                            for row in exonicInfoFile:
                                if row != getDataForThisChromOnly:
                                    continue
                                elif row == getDataForThisChromOnly:
                                    hit_b =1 
                                    break 
                            #The exonic info for the chromo is now at the following line:
                            if hit_b ==1 :
                                XexonicChromo = exonicInfoFile.next()
                        
                        else: #if no exon-info file was provided, we use 0:
                            
                            print("OBS: no file containing exonic info was provided, so exonic status is set to 0 throughout!")

                            for i in range(startAtPosition, endAtPosition +1):
                                XexonicChromo += str(0)
                        
                        XchrExonicAllDict[currChromo] = ''
                        
            



                elif currChromo == getDataForThisChromOnly and accumulatedLength <= endAtPosition:
    
                    accumulatedLength += len(v)
                    
                    if accumulatedLength >= startAtPosition: 
                        
                        XchrAllDict[currChromo] += v
                        lenXchrAll += len(v)
                        
                        XchrExonicAllDict[currChromo] += XexonicChromo[(accumulatedLength - len(v)):accumulatedLength]    
                    
                    if lineCnt%1000000 == 0:
                        print("Have read %d lines" % lineCnt)
                    
                    lineCnt += 1
                
        
        #Get length of exonic-info read in:
        lenXchrExonicAll = 0
        for chromo in XchrExonicAllDict.keys():
            
            lenXchrExonicAll += len(XchrExonicAllDict[chromo])
            
        print("Chromosomes: ", XchrAllDict.keys() )   
        print("Length genome sequence read in: %d" % lenXchrAll)
        print("Length of exonic-info sequence read in:%d" % lenXchrExonicAll)
        if lenXchrAll != lenXchrExonicAll:
            raw_input("Warning: lengths of exonic info and dna-seq differ!")

        
         
        #Not all letters are ACGT! We read the seq's in to dict's holding only ACGTacgt's:
        for chromo in XchrAllDict.keys():
            
            print("Now at chromosome: %s" % chromo)
            
            Xchr = ''
            XchrRepeat = []
            XchrExonic = []
            
            lenXchrAll = len(XchrAllDict[chromo])
            
            print("Length of this genome sequence:%d" % lenXchrAll)
            
            trailing_b = 0
            headingNs = 0
            trailingNs = 0
            for i in range(lenXchrAll):
                
                try:
                    if XchrAllDict[chromo][i] == 'A':            
                        Xchr += 'A'
                        XchrRepeat.append(0)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'T':     
                        Xchr += 'T'
                        XchrRepeat.append(0)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'C':    
                        Xchr += 'C'
                        XchrRepeat.append(0)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'G':    
                        Xchr += 'G' 
                        XchrRepeat.append(0)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'a':            
                        Xchr += 'A'
                        XchrRepeat.append(1)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 't':     
                        Xchr += 'T'
                        XchrRepeat.append(1)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'c':    
                        Xchr += 'C'
                        XchrRepeat.append(1)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    elif XchrAllDict[chromo][i] == 'g':    
                        Xchr += 'G' 
                        XchrRepeat.append(1)
                        XchrExonic.append(XchrExonicAllDict[chromo][i])
                        trailing_b = 1
                    else:
                        if XchrAllDict[chromo][i] == 'N':   
                            if trailing_b == 1:
                                trailingNs += 1
                            else:
                                headingNs += 1
                except IndexError:
                    print("Letter is %s so not ACGTacgt at: %d" % (XchrAllDict[chromo][i], i))
                    print("... or exonic info at %d is not found; exonic info file is %s" % (i,exonicInfoBinaryFileName ))
            XchrDict[chromo] = Xchr
            XchrRepeatDict[chromo] = XchrRepeat
            XchrExonicDict[chromo] = XchrExonic
            XchrNsDict[chromo] = [headingNs, trailingNs]
            
            lenXchr = len(XchrDict[chromo])  
#            print("S .. ?", lenXchr)
 

        return XchrAllDict, XchrDict, XchrRepeatDict, XchrExonicDict, XchrNsDict




def encodeGenome(fileName, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 10, 
               startAtPosition = 0,
               endAtPosition = int(1e26), #some gigantic number
               outputEncoded_b = 1,
               outputEncodedOneHot_b = 1,
               outputEncodedInt_b = 0,
               outputEncodedType = 'int8', 
               outputAsDict_b = 0,
               outputGenomeString_b =0,
               randomChromo_b = 0, 
               avoidChromo = []):
    ''' Reads in genome in fasta format by encoding the letters as 
    defined by codeA, codeT, codeC and codeG. 

    Input:
    chromoNameBound: allows to set a bound on the names of the chromosomes for 
    which data are read in (eg with chromoNameBound = 5 data for chr17 will be 
    read in but not for chr17_extra_assembly_information_nn)
    randomChromo_b: if 1 only dat afrom one randomly chosen chromosome will
    be read in. To avoid one or more chromosomes to be chosen provide a 
    "negative list", avoidChromo. 
    
    
    Outputs: structure of the output is
    dna-sequence content, dna-sequence content w repeat masking, repeat-information content, exonic-information content 
    
    and if output as dictionaries(mapping chromo to:)
    dna-sequence content, dna-sequence content w repeat masking, repeat-information content, exonic-information content, heading-trailing N'c content

    
    Inputs: (see also fct readGenome)
    
    outputAsDict_b: if 1 the dna-sequence content of the output consts in 
    a dictionary mapping each chromosome to its sequence; if 0 the complete 
    dna-sequence concatenated from the choromosomes' seqeuences is output
    
    outputEncoded_b: if 0 the dna-sequence content output consists of an array 
    of encoded letters in the genome's sequence order (or a dictionary mapping 
    each chromosome to that array); if = 1 the letters (alphabet
    ATCG) are "one-hot" encoded as follows:
    
    codeA = [1,0,0,0]
    codeC = [0,1,0,0]
    codeG = [0,0,1,0]
    codeT = [0,0,0,1]
    
    This conversion is case insensitive (ie repeat masking is not encoded).
    
    outputGenomeString_b: if outputEncoded_b == 1 the genome string will be 
    output (as third part of output tuple) when setting this parameter 
    (outputGenomeString_b) to 1.
    
    The output's repeat-information consists in a 0/1 indicator for each 
    position in the genome (sharing index with the dna-sequence content); if
    a letter is lower case, it is regarded as belonging to a repeat (see 
    ftp://ftp.ncbi.nlm.nih.gov/genomes/genbank/README.txt), and a 1 is recorded
    at that position; else a 0 is recorded (for not belonging to a repeat).
    
    The output's exonic-information consists in a 0/1/2 indicator (or another triple
    for annotating intergenic/exonic/intronic) for each position in the genome 
    (sharing index with the dna-sequence content).
    
    '''
    
    XexonicChromo = '' #to contain the full sequence of all 0/1 exonic indicators for (a desired part of) a chromo
    Xall = '' #to contain the full sequence of all letters in the read-in genome
    XexonicAll = '' #to contain the full sequence of all 0/1 exonic indicators in the read-in genome
    X = '' #to only contain the ACGT's of the genome
    Xrepeat = [] #to contain the indicator of a position being in a repeat or not
    Xexonic = [] #to contain the indicator of a position being exonic or not
    Xchr = '' #to only contain the ACGT's in one chromosome of the read-in genome
    XchrExonic = '' #to only contain the exonic-info in one chromosome of the read-in genome
    XchrRepeat = [] #to contain the indicator of a position being in a repeat or not
    XchrExonic = [] #to contain the indicator of a position being exonic or not
    XchrAllDict = {} #dict to map: each chromosome to its sequence
    XchrExonicAllDict = {} #dict to map: each chromosome to its exonic-info sequence
    XchrDict = {} #dict to map: each chromosome to list [its length, sequence(only ACGT's)]
    XchrRepeatDict = {} #repeat info corr to XchrDict
    XchrExonicDict = {} #exonic info corr to XchrDict
    XchrNsDict = {} #to collect the number of heading,trailing ends for each chromo
    lineCnt = 0
    lenXall = 0
    lenX = 0
    lenXchrAll = 0
    lenXchr = 0
    
    print("Reading in genome data ... ")
    print("Only considering data following fasta header lines (: chromo names \n for eucaryots) of length < %d" %  chromoNameBound)

    
    
    if outputAsDict_b == 0:
        
        #Read in data from genome
        Xall, X, Xrepeat, Xexonic = readGenome(fileName = fileName, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = outputAsDict_b,
               outputGenomeString_b = outputGenomeString_b,
               randomChromo_b = randomChromo_b, 
               avoidChromo = avoidChromo)
        
        
                
        #If desired the letters will be "one-hot" encoded:
        lenX = len(X)
        print("Length genome sequence, only ACGT's:%d" % lenX)
        
        if outputEncoded_b == 1:
            
            if outputEncodedOneHot_b == 1:
    
                Xenc = np.zeros(shape = (lenX,4), dtype = outputEncodedType)
                XencRepeat = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                XencExonic = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                
                for i in range(lenX):
            
                    try:
                        XencRepeat[i] = Xrepeat[i]
                        try:
                            XencExonic[i] = int(Xexonic[i])
                        except ValueError:
                            print("At pos %d set exonic info to 2 (other)" % i)
                            XencExonic[i] = 2
                        
                        if X[i] == 'A':            
                            Xenc[i] = codeA_asArray
                        elif X[i] == 'T':     
                            Xenc[i] = codeT_asArray
                        elif X[i] == 'C':    
                            Xenc[i] = codeC_asArray
                        elif X[i] == 'G':    
                            Xenc[i] = codeG_asArray
                    except IndexError:
                        print("Letter is: %s so not ACGTacgt at: %d" % (X[i],i))
                
                if outputGenomeString_b == 1:
                    return Xenc, XencRepeat, XencExonic, X

                else:
                    return Xenc, XencRepeat, XencExonic
                
                
            
            elif outputEncodedInt_b == 1:
    
                Xenc = np.zeros(shape = lenX, dtype = outputEncodedType)
                XencRepeat = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                XencExonic = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                
                for i in range(lenX):
            
                    try:
                        XencRepeat[i] = Xrepeat[i]
                        XencExonic[i] = int(Xexonic[i])
                        
                        if X[i] == 'A':            
                            Xenc[i] = codeA_asInt
                        elif X[i] == 'T':     
                            Xenc[i] = codeT_asInt
                        elif X[i] == 'C':    
                            Xenc[i] = codeC_asInt
                        elif X[i] == 'G':    
                            Xenc[i] = codeG_asInt
                    except IndexError:
                        print("Letter is: %s so not ACGTacgt at: %d" % (X[i],i))
                 
                if outputGenomeString_b == 1:
                    return Xenc, XencRepeat, XencExonic, X
                    
                else:
                    return Xenc, XencRepeat, XencExonic
        
        
        else:# outputEncoded_b == 0:
    
            return X, Xrepeat, Xexonic
                                                
        
        
    #read in data for each chromosome and record it all in a dictionary:
    else: #outputAsDict_b != 0:
        
        
        #Read in data from genome
        XchrAllDict, XchrDict, XchrRepeatDict, XchrExonicDict, XchrNsDict = readGenome(fileName = fileName, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = outputAsDict_b,
               outputGenomeString_b = outputGenomeString_b,
               randomChromo_b = randomChromo_b, 
               avoidChromo = avoidChromo)


        #If desired the letters will be "one-hot" encoded:
        if outputEncoded_b == 1:
    #        XencAllDict = {}
            XencDict = {}
            XencRepeatDict = {}
            XencExonicDict = {}
            
            if outputEncodedOneHot_b == 1:
            
                for chromo in XchrDict.keys():
                    
        #            lenXchrAll = XchrAll[chromo][0]
                    lenXchr = len(XchrDict[chromo])   
                    print("Soes .. ??", lenXchr)
                    XencDict[chromo] = np.zeros(shape = (lenXchr,4), dtype = outputEncodedType)
                    XencRepeatDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    XencExonicDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    
                    for i in range(lenXchr):
                
                        try:                            
    
                            XencRepeatDict[chromo][i] = XchrRepeatDict[chromo][i]
                            XencExonicDict[chromo][i] = int(XchrExonicDict[chromo][i])
                            
                            if XchrDict[chromo][i] == 'A':            
                                XencDict[chromo][i] = codeA_asArray
                            elif XchrDict[chromo][i] == 'T':     
                                XencDict[chromo][i] = codeT_asArray
                            elif XchrDict[chromo][i] == 'C':    
                                XencDict[chromo][i] = codeC_asArray
                            elif XchrDict[chromo][i] == 'G':    
                                XencDict[chromo][i] = codeG_asArray
                        except IndexError:
                            print("Letter is %s so not ACGTacgt at: %d" % (XchrDict[chromo][i], i))
                            print("... or repeat or exonic info is not found at %d" % i)
                            
            elif outputEncodedInt_b == 1:
                
                for chromo in XchrDict.keys():
                    
        #            lenXchrAll = XchrAll[chromo][0]
                    lenXchr = len(XchrDict[chromo])   
                    XencDict[chromo] = np.zeros(shape = lenXchr, dtype = outputEncodedType)
                    XencRepeatDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    XencExonicDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    
                    for i in range(lenXchr):
                
                        try:                            
    
                            XencRepeatDict[chromo][i] = XchrRepeatDict[chromo][i]
                            XencExonicDict[chromo][i] = int(XchrExonicDict[chromo][i])
                            
                            if XchrDict[chromo][i] == 'A':            
                                XencDict[chromo][i] = codeA_asInt
                            elif XchrDict[chromo][i] == 'T':     
                                XencDict[chromo][i] = codeT_asInt
                            elif XchrDict[chromo][i] == 'C':    
                                XencDict[chromo][i] = codeC_asInt
                            elif XchrDict[chromo][i] == 'G':    
                                XencDict[chromo][i] = codeG_asInt
                                
                        except IndexError:
                            print("Letter not ACGTacgt at: %d" % i)
        
        
        #Structure final output
        if outputEncoded_b == 1:
            
            if outputGenomeString_b == 1:
                return XencDict, XencRepeatDict, XencExonicDict, XchrDict, XchrNsDict
            else:
                return XencDict, XencRepeatDict, XencExonicDict
        
        else: #if outputEncoded_b == 0:
        
            return XchrDict, XchrRepeatDict, XchrExonicDict






def genomeStats(XencDict, XencRepeatDict, XencExonicDict):
    
    #Repeat stats:
    nrOfRepeatsAll = 0
    totalLength = 0
    repeatsFracDict = {}
    for chromo in XencRepeatDict:
        
        lChromo = XencRepeatDict[chromo].shape[0]
        nrOfRepeatsChromo = np.sum(XencRepeatDict[chromo])        
        
        if lChromo > 0:
            fractionRepeats = float(nrOfRepeatsChromo)/lChromo
        else:
            print("Warning: length of chromo %s is 0!" %  chromo)
            fractionRepeats = 0
        repeatsFracDict[chromo] = fractionRepeats
     
        print("Chromo: %s repeat fraction: %f " % (chromo, fractionRepeats))
        
        totalLength += lChromo
        nrOfRepeatsAll += nrOfRepeatsChromo
        
    fractionRepeats = float(nrOfRepeatsAll)/totalLength
    repeatsFracDict['all'] = fractionRepeats
    print("All chromos, repeat fraction: %f " % (fractionRepeats)   )
     
     
    #Exonic stats:
    nrOfExonicsAll = 0
    totalLength = 0
    exonicsFracDict = {}
    for chromo in XencExonicDict:
        
        lChromo = XencExonicDict[chromo].shape[0]
        nrOfExonicsChromo = np.sum(XencExonicDict[chromo])        
        
        if lChromo > 0:
            fractionExonics = float(nrOfExonicsChromo)/lChromo
        else:
            print("Warning: length of chromo %s is 0!" %  chromo)
            fractionExonics = 0
            
        exonicsFracDict[chromo] =  fractionExonics
     
        print("Chromo: %s exonic fraction: %f " % (chromo, fractionExonics))
        
        totalLength += lChromo
        nrOfExonicsAll += nrOfExonicsChromo
    
    fractionExonics = float(nrOfExonicsAll)/totalLength
    exonicsFracDict['all'] = fractionExonics
    print("All chromos, exonic fraction: %f " % (fractionExonics)   )
     
    return repeatsFracDict, exonicsFracDict
     

def generateExonicInfoFromFile(inputFilename, genomeFilename, chromoNameBound = 20, outputFilename = 'test', onlyCDS_b = 1 ):
    '''Take UCSC-file of exonic info and generate a string of 0/1's, 0 for non-exonic
    1 for exonic, and output the string to file. the inf will be separated by
    chromo names as in fasta format (ie like > chrX).
    
    Format of inputFile:
     
        #name	         chrom	cdsStart	cdsEnd	exonCount 	exonStarts     	exonEnds
        uc001aaa.3 	chr1	     11873	11873	3	         11873,12612,13220,	12227,12721,14409,
        
    
    genome file: must contain the corresponding genome assembly (fasta)   
    
    chromoNameBound: only info for chromosome having names of length less than this integer are considered. 
    '''

    infoDict = {}
    
    
    #First Loop through the genome-file to find the diff chromo's and their lengths:            
    lenChr = 0
    chromoDict = {}
    currChromo = ''
    for line in open(genomeFilename):
        
        v = line.strip()
        
        #skip line starting with a '>'; these header lines will preceed the dna-seq
        #or the dna-seq for each chromosome 
        if v[0] == '>' and len(v[1:]) < chromoNameBound:
            
            if currChromo != '':
                chromoDict[currChromo] = lenChr
            
            currChromo = v[1:]
            print("Found data for this chromosome: ", currChromo)
            lenChr = 0
                                                            
        else:
            
            lenChr += len(v)

    #record for lastly passed chromo too:
    chromoDict[currChromo] = lenChr
            
    print("Chromos considered: ", chromoDict.keys())
    
    #flush the output file for contents:
    writeToFile = open(outputFilename, 'w')
    writeToFile.close()
    #and open in append mode:
    writeToFile = open(outputFilename, 'a')


#    maxEndPositionCurrent = 0
    included_b = 0
    with open(inputFilename) as of:
        
        lineCnt = 0
        nowAtChromo =  ''
        for line in of:
            
            if lineCnt > 0:
                
                line = line.split()
                
#                print line[:2]
                
                if nowAtChromo == line[1] and included_b == 1:
                    
                    exonCnt = int(line[4])
                    exonStarts = line[5].split(",")
                    exonEnds = line[6].split(",")
                    #remove last entry of start and end, since always a blank:
                    exonStarts.pop(exonCnt)
                    exonEnds.pop(exonCnt)
                    exonStarts = map(int, exonStarts)
                    exonEnds = map(int, exonEnds)
#                    for i in range(exonCnt):
#                        exonStarts[i] = int(exonStarts[i])
#                        exonEnds[i] = int(exonEnds[i])
                    
#                    print maxEndPositionCurrent, exonEnds[exonCnt - 1]
#                    for i in range(maxEndPositionCurrent, exonEnds[exonCnt - 1]):
#                        
#                        infoDict[nowAtChromo].append(0)
                    
#                    maxEndPositionCurrent =  max(maxEndPositionCurrent, exonEnds[exonCnt - 1])
                    
                    #If desired to remove UTRs and pseudo-genes, replace exonStart/End by CDS start/End.
                    if onlyCDS_b == 1:
                        
                        exonStarts[0] =  int(line[2]) #CDS start
                        exonEnds[exonCnt -1] =  int(line[3]) #CDS end
                        
                        
                    #Replace the 0's by 1's at exonic positions
                    for j in range(exonCnt):
                        
                        for i in range(exonStarts[j], exonEnds[j]):
                            
                            infoDict[nowAtChromo][i] = 1  
                            
#                    print 1, lineCnt, len(infoDict[nowAtChromo])
                            
        
                    
                else:

                    #write lines to file when ready
                    if lineCnt > 1  and included_b == 1:
                        writeToFile.write('>' + nowAtChromo)
                        writeToFile.write("\n")
                        #convert the list of 0/1's to string of 0/1's
                        binString = ''
                        for i in range(len(infoDict[nowAtChromo])):
                            binString = binString +  str(infoDict[nowAtChromo][i])
                        writeToFile.write(binString)
                        writeToFile.write("\n")
                    
                    nowAtChromo = line[1]
                    
                    if chromoDict.has_key(nowAtChromo):
                        
                        print("Now at chromo: ", nowAtChromo)
                        lenChr = chromoDict[nowAtChromo]
                        #initially, set the exonic info to 0 
                        infoDict[nowAtChromo] = [0 for i in range(lenChr-1)]
                        included_b = 1
                    
                        exonCnt = int(line[4])
                        exonStarts = line[5].split(",")
                        exonEnds = line[6].split(",")
                        #remove last entry of start and end, since always a blank:
                        exonStarts.pop(exonCnt)
                        exonEnds.pop(exonCnt)
                        exonStarts = map(int, exonStarts)
                        exonEnds = map(int, exonEnds)
    #                    for i in range(exonCnt):
    #                        exonStarts[i] = int(exonStarts[i])
    #                        exonEnds[i] = int(exonEnds[i])                 
                                            
    #                    maxEndPositionCurrent =  exonEnds[exonCnt - 1]
    #                    for i in range(maxEndPositionCurrent):
    #                        
    #                        infoDict[nowAtChromo].append(0)
                        
                        #If desired to remove UTRs and pseudo-genes, replace exonStart/End by CDS start/End.
                        if onlyCDS_b == 1:
                            
                            exonStarts[0] =  int(line[2]) #CDS start
                            exonEnds[exonCnt -1] =  int(line[3]) #CDS end
                            
                        #Replace the 0's by 1's at exonic positions
                        for j in range(exonCnt):
                            
                            for i in range(exonStarts[j], exonEnds[j]):
                                
                                infoDict[nowAtChromo][i] = 1    

                    else:
                        
                        included_b = 0
                            
#                    print 0, lineCnt, len(infoDict[nowAtChromo])
                    
            lineCnt += 1
            
          
    #Write records to file for last chromo passed:        
    if lineCnt > 1 and included_b == 1:
        writeToFile.write('>' + nowAtChromo)
        writeToFile.write("\n")
        #convert the list of 0/1's to string of 0/1's
        binString = ''
        for i in range(len(infoDict[nowAtChromo])):
            binString = binString +  str(infoDict[nowAtChromo][i])
        writeToFile.write(binString)
        writeToFile.write("\n")
    
    writeToFile.close()
    
    return infoDict



#Mangler: augment ... delen
def getAllSamplesFromGenome(genomeData, 
                            labelsCodetype = 0,
                            outputEncodedOneHot_b = 1,
                            outputEncodedInt_b = 0,
                            outputEncodedType = 'int8',
                            convertToPict_b = 0,
                            flankSize = 50, 
                            shuffle_b = 0, 
                            inner_b = 1, 
                            shuffleLength = 5,
                            augmentWithRevComplementary_b = 0, 
                            cutDownNrSamplesTo = 1e26):
    '''Get all "samples" of flanks and corr midpoints moving across a genome.
    Input:
    genomeData: tuple dna-sequence, repeat info, exonic info as returned be readGenome
    '''
    
    genomeSeq, repeatInfoSeq, exonicInfoSeq =  genomeData

    #First sample is at idx flankSize, last at len(genomeData) - flankSize: 
    nrSamples = len(genomeSeq) - 2*flankSize -1

    nrSamples = min(cutDownNrSamplesTo, nrSamples)

#    firstSampleAt = flankSize
#    lastSampleAt = len(genomeData) - flankSize
 
    #Set a labels-shape depending on the labelsCodetype:
    if labelsCodetype == 0:
        
        labelShape = 4
    
    elif labelsCodetype == 1 or labelsCodetype == -1 or labelsCodetype == 3:
        
        labelShape = 2 
        
    elif labelsCodetype == 2:
        
        labelShape = 3
        
    
    if outputEncodedOneHot_b == 1:

        try:
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            
    elif outputEncodedInt_b == 1:

        try:
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
   
            
            
#    print "X shape", X.shape
#    print "Y shape", Y.shape
    
    #Loop along the genome and record the flanks and midpoints:
    for i in range(nrSamples):
        
        #left flank:
#        for j in range(flankSize):
#            X[i][j] = genomeSeq[i + j] #i + j = i + firstSampleAt - flankSize + j
        X[i][:flankSize] = genomeSeq[i:(i + flankSize)] #i + j = i + firstSampleAt - flankSize + j

        #right flank:
#        for j in range(flankSize + 1, 2*flankSize +1):
#            X[i][j-1] = genomeSeq[i + j]
        X[i][flankSize:] = genomeSeq[(i + flankSize +1):(i + 2*flankSize + 1)]
            
        if shuffle_b > 0.5:
#                print "I shuffle ..."
#            print v2
            if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            else:
                shuffle(X[i])          
   
        #labels at midpoint:
        R[i] = repeatInfoSeq[i + flankSize]
        
        if labelsCodetype == 0:
            Y[i] = genomeSeq[i + flankSize]
        elif labelsCodetype == 1:
            Y[i] = basePair(genomeSeq[i + flankSize])
        elif labelsCodetype == -1:
            Y[i] = basePairType(genomeSeq[i + flankSize])
        elif labelsCodetype == 2:
            if exonicInfoSeq[i + flankSize] == 1:
                Y[i] = exonicInd
            elif repeatInfoSeq[i + flankSize] == 1:
                Y[i] = repeatInd
            else:
                Y[i] = otherInd
        elif labelsCodetype == 3: #repeat or not?
            if repeatInfoSeq[i + flankSize] == 1:
                Y[i] = repeatBinInd
            else:
                Y[i] = notRepeatBinInd
        
        
    if outputEncodedOneHot_b == 1 and convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        
    return X, Y, R



def genSamples_I(nrSamples,
               outDtype = 'float32',
               genRandomSamples_b = 1,
                     outputEncodedOneHot_b = 1,
                     labelsCodetype = 0,
                     outputEncodedInt_b = 0,
                     convertToPict_b = 0,
                     flankSize = 50, 
                     shuffle_b = 0, 
                     inner_b = 1, 
                     shuffleLength = 5, 
                     getFrq_b = 0,
                     augmentWithRevComplementary_b = 0, 
                     fromGenome_b = 0,
                     startAtPosition = 0,
                     endAtPosition = int(1e26), #some gigantic number
                     genomeFileName = '',
                     exonicInfoBinaryFileName = '',
                     onlyOneRandomChromo_b = 0,
#                     randomChromo_b = 0, 
                     avoidChromo = [],
                     inclFrqModel_b = 0,
                     frqModelDict = {},
                     flankSizeFrqModel = 4,
                     exclFrqModelFlanks_b = 0):
    '''Generate a set of nrSamples samples. This can be do either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.
    
    labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                    or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                    chosen code type (ie if 1 is used it is only the base pair at the given position which
                    is predicted). Pt only works with on-hot encoding.
    
    
    
    '''
    
    print("startAtPosition ", startAtPosition)
    print("endAtPosition ", endAtPosition)
    
    if nrSamples > endAtPosition - startAtPosition:
        nrSamples = endAtPosition - startAtPosition
        print("Nr of samples reduced to %d which equals the length of the interval from startAtPosition to endAtPosition!")
        
    if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
        
        flankSizeOut = flankSize - exclFrqModelFlanks_b*flankSizeFrqModel
        
    else:
        
        flankSizeOut = flankSize
        
        
    #Set a labels-shape depending on the labelsCodetype:
    if labelsCodetype == 0:
        
        labelShape = 4
    
    elif labelsCodetype == 1 or labelsCodetype == -1 or labelsCodetype == 3:
        
        labelShape = 2 
        
    elif labelsCodetype == 2:
        
        labelShape = 3

        
    if outputEncodedOneHot_b == 1:

        if inclFrqModel_b == 1:
            
            if augmentWithRevComplementary_b == 1:
            
                X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks               
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outDtype ) #to hold the labels

            else:
                
                X = np.zeros(shape = (nrSamples, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks          
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outDtype ) #to hold the labels
                
        else:

            if augmentWithRevComplementary_b == 1:
                
                X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outDtype ) #to hold the labels

            else:
                
                X = np.zeros(shape = (nrSamples, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outDtype ) #to hold the labels
                

    elif outputEncodedInt_b == 1:
        
        if augmentWithRevComplementary_b == 1:
            X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outDtype ) #to hold the flanks
            Y = np.zeros(shape = (2*nrSamples,1), dtype = outDtype ) #to hold the labels
        else:
            X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outDtype ) #to hold the flanks
            Y = np.zeros(shape = (nrSamples), dtype = outDtype ) #to hold the labels
    

    #set a random seed (obs: 2**32 -1 > length of human genome):
    np.random.seed(seed = np.random.randint(0,2**32 -1))
    

    #NOT IN WORKING ORDER
    if fromGenome_b == 0:
        
        i = 0
        for i in range(nrSamples):
            
            #generate random "letters", codeA, codeC, codeT, codeG
            for j in range(2*flankSize):
                X[i][j] = np.random.randint(0,4) 
                
            #generate random label:
            #NOT OK -- should retrun vector
            Y[i] = np.random.randint(0,labelShape)
    
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
                
                if outputEncodedOneHot_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementArray, xRev)
                    X[i + nrSamples] = xRevAug
                    if labelsCodetype == 0:
                        Y[i + nrSamples ] = complementArray(Y[i])
                    elif labelsCodetype == 1:
                        Y[i + nrSamples ] = Y[i]
                    elif labelsCodetype == -1:
                        Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                    elif labelsCodetype == 2:
                        Y[i + nrSamples ] = Y[i]
                    elif labelsCodetype == 3:
                        Y[i + nrSamples ] = Y[i]
                    
                elif outputEncodedInt_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementInt, xRev)
                    X[i + nrSamples] = xRevAug
                    Y[i + nrSamples ] = complementInt(Y[i])
                
            genomeSeqSource =  'Samples from randomly generated genome'

    elif fromGenome_b > 0.5:
        
        lGenome = 0
        m = 0
        while lGenome < nrSamples:

            #read in the genome sequence:
            if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
                genomeArray, repeatArray, exonicArray, genomeString = encodeGenome(fileName = genomeFileName, exonicInfoBinaryFileName = exonicInfoBinaryFileName, outputGenomeString_b = 1, startAtPosition = startAtPosition + m*nrSamples, endAtPosition = endAtPosition  + m*nrSamples, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 0, randomChromo_b = 0, avoidChromo = avoidChromo)
                lGenome = len(genomeArray)
                genomeSeqSource = 'Read data from whole genome (chromo\'s concatenated, if any)'
            elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
                genomeDictArray, repeatInfoDictArray, exonicInfoDictArray, genomeDictString = encodeGenome(fileName = genomeFileName, exonicInfoBinaryFileName = exonicInfoBinaryFileName, outputGenomeString_b = 1, startAtPosition = startAtPosition + m*nrSamples, endAtPosition = endAtPosition  + m*nrSamples, outputEncoded_b = 1, outputEncodedOneHot_b = outputEncodedOneHot_b, outputEncodedInt_b = outputEncodedInt_b, outputAsDict_b = 1, randomChromo_b = 1, avoidChromo = avoidChromo)
                if len(genomeDictArray.keys()) > 1:
                    print("Warning: more than one chromosome has been selected")
                chromo = genomeDictArray.keys()[0]
                genomeArray = genomeDictArray[chromo]
                repeatArray = repeatInfoDictArray[chromo]
                exonicArray = exonicInfoDictArray[chromo]
                genomeString = genomeDictString[chromo]
                lGenome = len(genomeArray)
                genomeSeqSource = chromo

            m += 1
            
            if lGenome < nrSamples:
                print("The length of the obtained genome-string is shorter than the desired nr of samples, so will try fetching the samples from the interval: ", [startAtPosition + m*nrSamples, endAtPosition  + m*nrSamples])
            

                
        print("lGenome: %d" % lGenome)

        
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site unless we want to simply get the nrSamples samples from a running window of
            #length 2*flankSize across the selected part of the genome:
            if genRandomSamples_b == 0:
                idx = flankSize + i
            else:
                idx = np.random.randint(flankSize+1,lGenome-flankSize) 
                if labelsCodetype == 0:
                    Y[i] = genomeArray[idx]
                elif labelsCodetype == 1:
                    Y[i] = basePair(genomeArray[idx])
                elif labelsCodetype == -1:
                    Y[i] = basePairType(genomeArray[idx])
                elif labelsCodetype == 2:
                    if exonicArray[idx] == 1:
                        Y[i] = exonicInd
                    elif repeatArray[idx] == 1:
                        Y[i] = repeatInd
                    else:
                        Y[i] = otherInd
                elif labelsCodetype == 3: #repeat or not?
                    if repeatArray[idx] == 1:
                        Y[i] = repeatBinInd
                    else:
                        Y[i] = notRepeatBinInd
#            print "idx:", idx
            
            #... and fetch the correspondning flanks:
            if inclFrqModel_b == 0:
                X[i][:flankSize] =  genomeArray[(idx - flankSize):idx] #left-hand flank
                X[i][flankSize:] =  genomeArray[(idx+1):(idx + 1 + flankSize)] #right-hand flank
                
            if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
                
                
                genString = genomeString[(idx - flankSizeFrqModel):idx] + genomeString[(idx + 1):(idx + 1 + flankSizeFrqModel)]
                try:
                    X[i][flankSizeOut] = frqModelDict[genString]
                except KeyError:
                    X[i][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                
                X[i][:flankSizeOut] =  genomeArray[(idx - flankSize):(idx-exclFrqModelFlanks_b*flankSizeFrqModel)] #left-hand flank
                X[i][(flankSizeOut+1):] =  genomeArray[(idx+exclFrqModelFlanks_b*flankSizeFrqModel+1):(idx + 1 + flankSize)] #right-hand flank
     
            
            if shuffle_b > 0.5:
#                print "I shuffle ..."
#            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])          
            
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
               
               if outputEncodedOneHot_b == 1:
                    
                    if inclFrqModel_b == 0:
                        
                        xRev = X[i][::-1]
                        xRevAug = map(complementArray, xRev)
                        X[i + nrSamples] = xRevAug
                        if labelsCodetype == 0:
                            Y[i + nrSamples ] = complementArray(Y[i])
                        elif labelsCodetype == 1:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == -1:
                            Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                        elif labelsCodetype == 2:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == 3:
                            Y[i + nrSamples ] = Y[i]
                    
                    if inclFrqModel_b == 1:
                        
                        revComplGenString = ''.join(map(complementLetter,genString))
                        try:
                            X[i + nrSamples][flankSizeOut] = frqModelDict[revComplGenString]
                        except KeyError:
                            X[i + nrSamples][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                        
                        X[i + nrSamples][:flankSizeOut] = map(complementArray, X[i + nrSamples][(flankSizeOut+1):])
                        X[i + nrSamples][(flankSizeOut+1):] =  map(complementArray, X[i + nrSamples][:flankSizeOut])
                        if labelsCodetype == 0:
                            Y[i + nrSamples ] = complementArray(Y[i])
                        elif labelsCodetype == 1:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == -1:
                            Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                        elif labelsCodetype == 2:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == 3:
                            Y[i + nrSamples ] = Y[i]
                    
              
               elif outputEncodedInt_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementInt, xRev)
                    X[i + nrSamples] = xRevAug
                    Y[i + nrSamples ] = complementInt(Y[i])
            
            
    if outputEncodedOneHot_b == 1 and convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        
    #Get frequnecies if desired:
    if getFrq_b > 0.5:
        
        cntA_flanks = 0
        cntT_flanks = 0
        cntC_flanks = 0
        cntG_flanks = 0          
        
        cntA_midpoints = 0
        cntT_midpoints = 0
        cntC_midpoints = 0
        cntG_midpoints = 0  
        
        if outputEncodedOneHot_b == 1:
            
            k = 0
            for i in range(nrSamples):
                
                for j in range(2*flankSize):
                    
                    k += 1
                    
                    if np.array_equal(X[i][j], codeA_asArray):
                        cntA_flanks +=1. 
                    elif np.array_equal(X[i][j], codeT_asArray):
                        cntT_flanks +=1. 
                    elif np.array_equal(X[i][j], codeC_asArray):
                        cntC_flanks +=1. 
                    elif np.array_equal(X[i][j], codeG_asArray):
                        cntG_flanks +=1.
                        
                if np.array_equal(Y[i], codeA_asArray):
                    cntA_midpoints +=1. 
                elif np.array_equal(Y[i], codeT_asArray):
                    cntT_midpoints +=1. 
                elif np.array_equal(Y[i], codeC_asArray):
                    cntC_midpoints +=1. 
                elif np.array_equal(Y[i], codeG_asArray):
                    cntG_midpoints +=1.
                    
                k += 1

        elif outputEncodedInt_b == 1:
            
            k = 0
            for i in range(nrSamples):
                
                for j in range(2*flankSize):
                    
                    k += 1
                    
                    if np.array_equal(X[i][j], codeA_asInt):
                        cntA_flanks +=1. 
                    elif np.array_equal(X[i][j], codeT_asInt):
                        cntT_flanks +=1. 
                    elif np.array_equal(X[i][j], codeC_asInt):
                        cntC_flanks +=1. 
                    elif np.array_equal(X[i][j], codeG_asInt):
                        cntG_flanks +=1.
                        
                if np.array_equal(Y[i], codeA_asInt):
                    cntA_midpoints +=1. 
                elif np.array_equal(Y[i], codeT_asInt):
                    cntT_midpoints +=1. 
                elif np.array_equal(Y[i], codeC_asInt):
                    cntC_midpoints +=1. 
                elif np.array_equal(Y[i], codeG_asInt):
                    cntG_midpoints +=1.
                    
                k += 1            
        
        
        print("Generated %d flank samples with %d labels." % (len(X), len(Y)))
        
        frqA_flanks = cntA_flanks/k
        frqT_flanks = cntT_flanks/k
        frqC_flanks = cntC_flanks/k
        frqG_flanks = cntG_flanks/k
                
        print("A frq, flanks: %f" % frqA_flanks)
        print("T frq, flanks: %f" % frqT_flanks)
        print("C frq, flanks: %f" % frqC_flanks)
        print("G frq, flanks: %f" % frqG_flanks)
        
        
        frqA_midpoints = cntA_midpoints/nrSamples
        frqT_midpoints = cntT_midpoints/nrSamples
        frqC_midpoints = cntC_midpoints/nrSamples
        frqG_midpoints = cntG_midpoints/nrSamples
                
        print("A frq, midpoints: %f" % frqA_midpoints)
        print("T frq, midpoints: %f" % frqT_midpoints)
        print("C frq, midpoints: %f" % frqC_midpoints)
        print("G frq, midpoints: %f" % frqG_midpoints)
                
    return X, Y, genomeSeqSource      
    



#labelsCodetype NOT ENCORPORATED
def genSamples_II(nrSamples,
                  labelSize = 2,
               outDtype = 'float32',
               genRandomSamples_b = 1,
                     convertToPict_b = 0,
                     flankSize = 50, 
                     shuffle_b = 0, 
                     inner_b = 1, 
                     shuffleLength = 5, 
                     getFrq_b = 0,
                     augmentWithRevComplementary_b = 0, 
                     fromGenome_b = 0,
                     startAtPosition = 0,
                     endAtPosition = int(1e26), #some gigantic number
                     genomeFileName = '',
                     onlyOneRandomChromo_b = 0,
#                     randomChromo_b = 0, 
                     avoidChromo = [],
                     frqModelDict = {},
                     flankSizeFrqModel = 4,
                     exclFrqModelFlanks_b = 0):
    '''Generate a set of nrSamples samples. This can be do either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.'''
    
    if nrSamples > endAtPosition - startAtPosition:
        nrSamples = endAtPosition - startAtPosition
        print("Nr of samples reduced to %d which equals the length of the interval from startAtPosition to endAtPosition!")

        
    flankSizeOut = flankSize - exclFrqModelFlanks_b*flankSizeFrqModel
        
            
    if augmentWithRevComplementary_b == 1:
        X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut ,4), dtype = outDtype ) #to hold the flanks
        Y = np.zeros(shape = (2*nrSamples, labelSize), dtype = outDtype ) #to hold the labels
    else:
        X = np.zeros(shape = (nrSamples, 2*flankSizeOut ,4), dtype = outDtype ) #to hold the flanks
        Y = np.zeros(shape = (nrSamples, labelSize), dtype = outDtype ) #to hold the labels
                
    

    #set a random seed bewteen 0 and 1e6:
    np.random.seed(seed = np.random.randint(0,1e6))
    

    if fromGenome_b == 0:
        
        i = 0
        for i in range(nrSamples):
            
            #generate random "letters", codeA, codeC, codeT, codeG
            for j in range(2*flankSize):
                X[i][j] = np.random.randint(0,4) 
                
            #generate random label:
            Y[i] = np.random.randint(0,4)
    
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
                
                xRev = X[i][::-1]
                xRevAug = map(complementArray, xRev)
                X[i + nrSamples] = xRevAug
                Y[i + nrSamples ] = complementArray(Y[i])
            
            genomeSeqSource =  'Samples from randomly generated genome'

    elif fromGenome_b > 0.5:

        #read in the genome sequence:
        if onlyOneRandomChromo_b == 0: #the whole genome seq will be read in (chromo's concatenated, if any)
            genomeArray, repeatInfoArray, genomeString = encodeGenome(genomeFileName, outputGenomeString_b = 1, startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputEncoded_b = 1, outputEncodedOneHot_b = 1, outputEncodedInt_b = 0, outputAsDict_b = 0, randomChromo_b = 0, avoidChromo = avoidChromo)
            lGenome = len(genomeArray)
            genomeSeqSource = 'Read data from whole genome (chromo\'s concatenated, if any)'
        elif onlyOneRandomChromo_b == 1: #only the genome seq for one randomly chosen chromo (not in avoidChromo's list) will be read in:
            genomeDictArray, repeatInfoDictArray, genomeDictString = encodeGenome(genomeFileName,  outputGenomeString_b = 1, startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputEncoded_b = 1, outputEncodedOneHot_b = 1, outputEncodedInt_b = 0, outputAsDict_b = 1, randomChromo_b = 1, avoidChromo = avoidChromo)
            if len(genomeDictArray.keys()) > 1:
                print("Warning: more than one chromosome has been selected")
            chromo = genomeDictArray.keys()[0]
            genomeArray = genomeDictArray[chromo]
            genomeString = genomeDictString[chromo]
            lGenome = len(genomeArray)
            genomeSeqSource = chromo
            
        print("lGenome: %d" % lGenome)

        
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site unless we want to simply get the nrSamples samples from a running window of
            #length 2*flankSize across the selected part of the genome:
            if genRandomSamples_b == 0:
                idx = flankSize + i
            else:
                idx = np.random.randint(flankSize+1,lGenome-flankSize) 
            label = genomeArray[idx]
#            print "idx:", idx
            
            #... and fetch the correspondning flanks:                
            genString = genomeString[(idx - flankSizeFrqModel):idx] + genomeString[(idx + 1):(idx + 1 + flankSizeFrqModel)]
            if len(genString) != 2*flankSizeFrqModel:
                print("Something's rotten, lgth of genString is: ", len(genString) )
            try:
                predIdx = np.argmax(frqModelDict[genString])
            except KeyError:
                print("KeyError when reading from frqModelDict, key: ", genString)
                predIdx = np.random.randint(4)

                
            X[i][:flankSizeOut] = genomeArray[(idx - flankSize):(idx - exclFrqModelFlanks_b*flankSizeFrqModel)] #left-hand flank
            X[i][flankSizeOut:] = genomeArray[(idx+exclFrqModelFlanks_b*flankSizeFrqModel+1):(idx + 1 + flankSize)] #right-hand flank
            
            #the output we want records if the frqModel's bid differs from the actual label or not;
            #either we just store this crude info (as a 1 or 0, resp.), or we store the actual dinucleo
            #but one-hot it (in which case the labelSize must be 16):
            if labelSize == 2:
                
                if np.max(label - alphabetOnehot[predIdx]) == 0:
                    Y[i] = np.array([1,0])
                else:
                    Y[i] = np.array([0,1])
                
            elif labelSize == 16:
                
                #the labels record the true letter (e.g A) followed by the frq-model's prediction (e.g.T), so
                #a dinucleotide (here AT)
                #generate a one-hot encoding for these dinucleos: 
                #the label will be a 16 long array of zeros except a 1 at one place;
                #AA will be [1,0,...], AC [0,1,0,...] ...
                #CA will be [0,0,0,0,1,0,...], CC [0,0,0,0,0,1,0,...]
                #etc:
                n1 = np.argmax(label)
                nonZeroIdx = 4*n1 + predIdx
                Y[i][nonZeroIdx] = 1
                
            else:
                print("Fatal error: label size must be either 2 or 16")
                return
    
    
            
            if shuffle_b > 0.5:
#                print "I shuffle ..."
#            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])          
            
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
              
                print("Fatal error: augmentWithRevComplementary_b is not in working order here!")
                return 

              
            
            
    if convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        
    #Get frequnecies if desired:
    if getFrq_b > 0.5:
        
        cntA_flanks = 0
        cntT_flanks = 0
        cntC_flanks = 0
        cntG_flanks = 0          
        
        cntA_midpoints = 0
        cntT_midpoints = 0
        cntC_midpoints = 0
        cntG_midpoints = 0  
        
            
        k = 0
        for i in range(nrSamples):
            
            for j in range(2*flankSize):
                
                k += 1
                
                if np.array_equal(X[i][j], codeA_asArray):
                    cntA_flanks +=1. 
                elif np.array_equal(X[i][j], codeT_asArray):
                    cntT_flanks +=1. 
                elif np.array_equal(X[i][j], codeC_asArray):
                    cntC_flanks +=1. 
                elif np.array_equal(X[i][j], codeG_asArray):
                    cntG_flanks +=1.
                    
            if np.array_equal(Y[i], codeA_asArray):
                cntA_midpoints +=1. 
            elif np.array_equal(Y[i], codeT_asArray):
                cntT_midpoints +=1. 
            elif np.array_equal(Y[i], codeC_asArray):
                cntC_midpoints +=1. 
            elif np.array_equal(Y[i], codeG_asArray):
                cntG_midpoints +=1.
                
            k += 1     
        
        
        print("Generated %d flank samples with %d labels." % (len(X), len(Y)))
        
        frqA_flanks = cntA_flanks/k
        frqT_flanks = cntT_flanks/k
        frqC_flanks = cntC_flanks/k
        frqG_flanks = cntG_flanks/k
                
        print("A frq, flanks: %f" % frqA_flanks)
        print("T frq, flanks: %f" % frqT_flanks)
        print("C frq, flanks: %f" % frqC_flanks)
        print("G frq, flanks: %f" % frqG_flanks)
        
        
        frqA_midpoints = cntA_midpoints/nrSamples
        frqT_midpoints = cntT_midpoints/nrSamples
        frqC_midpoints = cntC_midpoints/nrSamples
        frqG_midpoints = cntG_midpoints/nrSamples
                
        print("A frq, midpoints: %f" % frqA_midpoints)
        print("T frq, midpoints: %f" % frqT_midpoints)
        print("C frq, midpoints: %f" % frqC_midpoints)
        print("G frq, midpoints: %f" % frqG_midpoints)
                
    return X, Y, genomeSeqSource 




def genSamplesForDynamicSampling_I(nrSamples,
                                 genomeArray, 
                                 repeatArray,
                                 exonicArray,
                                 transformStyle_b = 0,
                                 X = 0,
                                 Y = 0,
                                 labelsCodetype = 0,
                                 fromGenome_b = 1, 
                                 inclFrqModel_b = 0,
                                 frqModelDict = {},
                                 flankSizeFrqModel = 4,
                                 exclFrqModelFlanks_b = 0,
                                 genomeString = '', 
                     outDtype = 'float32',
                     getOnlyRepeats_b = 0,
                     genRandomSamples_b = 1,
                     outputEncodedOneHot_b = 1,
                     outputEncodedInt_b = 0,
                     convertToPict_b = 0,
                     flankSize = 50, 
                     shuffle_b = 0, 
                     inner_b = 1, 
                     shuffleLength = 5, 
                     getFrq_b = 0,
                     augmentWithRevComplementary_b = 1
                     ):
    '''Generate a set of nrSamples samples. This can be done either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.
    
    labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1), base 
                pair type (purine/pyrimidine, -1) or exonic/repeat/other (2); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding.


'''
    
    lGenome = genomeArray.shape[0] #length of genome sequnce
    if nrSamples > lGenome:
        nrSamples = lGenome - 2*flankSize
        print("Nr of samples reduced to %d which equals the length of the interval from length of genome sequence less twice the flank size")


    if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
        
        flankSizeOut = flankSize - exclFrqModelFlanks_b*flankSizeFrqModel
        
    else:
        
        flankSizeOut = flankSize
        
        
    #Set a labels-shape depending on the labelsCodetype:
    if labelsCodetype == 0:
        
        labelShape = 4
    
    elif labelsCodetype == 1 or labelsCodetype == -1 or labelsCodetype == 3:
        
        labelShape = 2 
        
    elif labelsCodetype == 2:
        
        labelShape = 3
        

    if transformStyle_b == 0:
        
        if outputEncodedOneHot_b == 1:
            
            if inclFrqModel_b == 1:
                
                if augmentWithRevComplementary_b == 1:
                    X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
                    Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outDtype ) #to hold the labels

                else:
                    X = np.zeros(shape = (nrSamples, 2*flankSizeOut + 1 ,4), dtype = outDtype ) #to hold the flanks
                    Y = np.zeros(shape = (nrSamples, labelShape), dtype = outDtype ) #to hold the labels
    
            else:
    
                if augmentWithRevComplementary_b == 1:
                    X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                    Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outDtype ) #to hold the labels

                else:
                    X = np.zeros(shape = (nrSamples, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
                    Y = np.zeros(shape = (nrSamples, labelShape), dtype = outDtype ) #to hold the labels
                    
    
        elif outputEncodedInt_b == 1:
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples,1), dtype = outDtype ) #to hold the labels
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outDtype ) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outDtype ) #to hold the labels
    

    #set a random seed bewteen 0 and 1e6:
#    np.random.seed(seed = np.random.randint(0,1e6))
    
    #NOT IN WORING ORDER!!
    if fromGenome_b == 0:
        
        i = 0
        for i in range(nrSamples):
            
            #generate random "letters", codeA, codeC, codeT, codeG
            for j in range(2*flankSize):
                X[i][j] = np.random.randint(0,4) 
                
                
            #generate random label:
            #NOT OK: should give a vector not an int
            Y[i] = np.random.randint(0,labelShape)
    
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
                
                if outputEncodedOneHot_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementArray, xRev)
                    X[i + nrSamples] = xRevAug
                    if labelsCodetype == 0:
                        Y[i + nrSamples ] = complementArray(Y[i])
                    elif labelsCodetype == 1:
                        Y[i + nrSamples ] = Y[i]
                    elif labelsCodetype == -1:
                        Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                
#            #generate random label:
#            Y[i] = np.random.randint(0,4)
#    
#            #add reversed and complemented sequence: 
#            if augmentWithRevComplementary_b == 1:
#                
#                if outputEncodedOneHot_b == 1:
#                    xRev = X[i][::-1]
#                    xRevAug = map(complementArray, xRev)
#                    X[i + nrSamples] = xRevAug
#                    Y[i + nrSamples ] = complementArray(Y[i])
#                
                elif outputEncodedInt_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementInt, xRev)
                    X[i + nrSamples] = xRevAug
                    Y[i + nrSamples ] = complementInt(Y[i])


    elif fromGenome_b > 0.5:

        
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site unless we want to simply get the nrSamples samples from a running window of
            #length 2*flankSize across the selected part of the genome:
            if genRandomSamples_b == 0:
                idx = flankSize + i
            else:
                idx = np.random.randint(flankSize+1,lGenome-flankSize) 

            #If only repeat-positions are wanted:
            if getOnlyRepeats_b == 1:
                
                while repeatArray[idx] == 0:
                    
                    if genRandomSamples_b == 0:
                        idx += 1
                    else:
                        idx = np.random.randint(flankSize+1,lGenome-flankSize) 

            
            if labelsCodetype == 0:
                Y[i] = genomeArray[idx]
            elif labelsCodetype == 1:
                Y[i] = basePair(genomeArray[idx])
            elif labelsCodetype == -1:
                Y[i] = basePairType(genomeArray[idx])
            elif labelsCodetype == 2:
                if exonicArray[idx] == 1:
                    Y[i] = exonicInd
                elif repeatArray[idx] == 1:
                    Y[i] = repeatInd
                else:
                    Y[i] = otherInd
            elif labelsCodetype == 3: #repeat or not?
                if repeatArray[idx] == 1:
                    Y[i] = repeatBinInd
                else:
                    Y[i] = notRepeatBinInd
            
#            print "idx:", idx
                    
            
            #... and fetch the correspondning flanks:
            if inclFrqModel_b == 0:
                X[i][:flankSize] = genomeArray[(idx - flankSize):idx] #left-hand flank
                X[i][flankSize:] = genomeArray[(idx+1):(idx + 1 + flankSize)] #right-hand flank
            
            if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
                
                genString = genomeString[(idx - flankSizeFrqModel):idx] + genomeString[(idx + 1):(idx + 1 + flankSizeFrqModel)]
                if len(genString) != 2*flankSizeFrqModel:
                    print("Something's rotten, lgth of genString is: ", len(genString) )
                try:
                    X[i][flankSizeOut] = frqModelDict[genString]
                except KeyError:
                    prin( "KeyError when reading from frqModelDict, key: ", genString)
                    X[i][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                
                X[i][:flankSizeOut] = genomeArray[(idx - flankSize):(idx - exclFrqModelFlanks_b*flankSizeFrqModel)] #left-hand flank
                X[i][(flankSizeOut+1):] = genomeArray[(idx+exclFrqModelFlanks_b*flankSizeFrqModel+1):(idx + 1 + flankSize)] #right-hand flank
                
            
            if shuffle_b > 0.5:
#                print "I shuffle ..."
#            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])          
            
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
                
                if outputEncodedOneHot_b == 1:
                    
                    if inclFrqModel_b == 0:
                        xRev = X[i][::-1]
                        xRevAug = map(complementArray, xRev)
                        X[i + nrSamples] = xRevAug
                        if labelsCodetype == 0:
                            Y[i + nrSamples ] = complementArray(Y[i])
                        elif labelsCodetype == 1:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == -1:
                            Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                        elif labelsCodetype == 2:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == 3:
                            Y[i + nrSamples ] = Y[i]
                            
                    if inclFrqModel_b == 1:
                        
                        revComplGenString = ''.join(map(complementLetter,genString))
                        try:
                            X[i + nrSamples][flankSizeOut] = frqModelDict[revComplGenString]
                        except KeyError:
                            X[i + nrSamples][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                
                        
                        X[i + nrSamples][:flankSizeOut] = map(complementArray, X[i + nrSamples][(flankSizeOut+1):])
                        X[i + nrSamples][(flankSizeOut+1):] =  map(complementArray, X[i + nrSamples][:flankSizeOut])
                        if labelsCodetype == 0:
                            Y[i + nrSamples ] = complementArray(Y[i])
                        elif labelsCodetype == 1:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == -1:
                            Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
                        elif labelsCodetype == 2:
                            Y[i + nrSamples ] = Y[i]
                        elif labelsCodetype == 3:
                            Y[i + nrSamples ] = Y[i]
                
                elif outputEncodedInt_b == 1:
                    xRev = X[i][::-1]
                    xRevAug = map(complementInt, xRev)
                    X[i + nrSamples] = xRevAug
                    Y[i + nrSamples ] = complementInt(Y[i])
            
            
    if outputEncodedOneHot_b == 1 and convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        

     
#    if transformStyle_b == 0:  
         
    return X, Y  
    



#labelsCodetype NOT ENCORPORATED
def genSamplesForDynamicSampling_II(nrSamples,
                                 genomeArray, 
                                 transformStyle_b = 0,
                                 X = 0,
                                 Y = 0,
                                 fromGenome_b = 1, 
#                                 inclFrqModel_b = 0,
                                 frqModelDict = {},
                                 flankSizeFrqModel = 4,
                                 exclFrqModelFlanks_b = 0,
                                 genomeString = '', 
                     outDtype = 'float32',
                     genRandomSamples_b = 1,
#                     outputEncodedOneHot_b = 1,
                     labelSize = 2,
#                     outputEncodedInt_b = 0,
                     convertToPict_b = 0,
                     flankSize = 50, 
                     shuffle_b = 0, 
                     inner_b = 1, 
                     shuffleLength = 5, 
                     getFrq_b = 0,
                     augmentWithRevComplementary_b = 1
                     ):
    '''Generate a set of nrSamples samples. This can be do either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.'''
    
    lGenome = genomeArray.shape[0] #length of genome sequnce
    if nrSamples > lGenome:
        nrSamples = lGenome - 2*flankSize
        print("Nr of samples reduced to %d which equals the length of the interval from length of genome sequence less twice the flank size")

        
    flankSizeOut = flankSize - exclFrqModelFlanks_b*flankSizeFrqModel
        


    if transformStyle_b == 0:
        
                            
        if augmentWithRevComplementary_b == 1:
            X = np.zeros(shape = (2*nrSamples, 2*flankSizeOut ,4), dtype = outDtype ) #to hold the flanks
            Y = np.zeros(shape = (2*nrSamples, labelSize), dtype = outDtype ) #to hold the labels
        else:
            X = np.zeros(shape = (nrSamples, 2*flankSizeOut,4), dtype = outDtype ) #to hold the flanks
            Y = np.zeros(shape = (nrSamples, labelSize), dtype = outDtype ) #to hold the label


    #set a random seed bewteen 0 and 1e6:
#    np.random.seed(seed = np.random.randint(0,1e6))
    

    #Not in working order!!:
    if fromGenome_b == 0:
        
        print("Fatal error: fromGenome_b =0 isn't in working order")
        return
        
#        i = 0
#        for i in range(nrSamples):
#            
#            #generate random "letters", codeA, codeC, codeT, codeG
#            for j in range(2*flankSize):
#                X[i][j] = np.random.randint(0,4) 
#                
#            #generate random label:
#            Y[i] = np.random.randint(0,4)
#    
#            #add reversed and complemented sequence: 
#            if augmentWithRevComplementary_b == 1:
#                
#                if outputEncodedOneHot_b == 1:
#                    xRev = X[i][::-1]
#                    xRevAug = map(complementArray, xRev)
#                    X[i + nrSamples] = xRevAug
#                    Y[i + nrSamples ] = complementArray(Y[i])
#                elif outputEncodedInt_b == 1:
#                    xRev = X[i][::-1]
#                    xRevAug = map(complementInt, xRev)
#                    X[i + nrSamples] = xRevAug
#                    Y[i + nrSamples ] = complementInt(Y[i])


    elif fromGenome_b > 0.5:

        
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site unless we want to simply get the nrSamples samples from a running window of
            #length 2*flankSize across the selected part of the genome:
            if genRandomSamples_b == 0:
                idx = flankSize + i
            else:
                idx = np.random.randint(flankSize+1,lGenome-flankSize) 
            
            label = genomeArray[idx]
#            print "idx:", idx
            
                            
            genString = genomeString[(idx - flankSizeFrqModel):idx] + genomeString[(idx + 1):(idx + 1 + flankSizeFrqModel)]
            if len(genString) != 2*flankSizeFrqModel:
                print("Something's rotten, lgth of genString is: ", len(genString) )
            try:
                predIdx = np.argmax(frqModelDict[genString])
            except KeyError:
                print("KeyError when reading from frqModelDict, key: ", genString)
                predIdx = np.random.randint(4)
            
            X[i][:flankSizeOut] = genomeArray[(idx - flankSize):(idx - exclFrqModelFlanks_b*flankSizeFrqModel)] #left-hand flank
            X[i][flankSizeOut:] = genomeArray[(idx+exclFrqModelFlanks_b*flankSizeFrqModel+1):(idx + 1 + flankSize)] #right-hand flank
            
            #the output we want records if the frqModel's bid differs from the actual label or not;
            #either we just store this crude info (as a 1 or 0, resp.), or we store the actual dinucleo
            #but one-hot it (in which case the labelSize must be 16):
            if labelSize == 2:
                
                if np.max(label - alphabetOnehot[predIdx]) == 0:
                    Y[i] = np.array([1,0])
                else:
                    Y[i] = np.array([0,1])
                
            elif labelSize == 16:
                
                #the labels record the true letter (e.g A) followed by the frq-model's prediction (e.g.T), so
                #a dinucleotide (here AT)
                #generate a one-hot encoding for these dinucleos: 
                #the label will be a 16 long array of zeros except a 1 at one place;
                #AA will be [1,0,...], AC [0,1,0,...] ...
                #CA will be [0,0,0,0,1,0,...], CC [0,0,0,0,0,1,0,...]
                #etc:
                n1 = np.argmax(label)
                nonZeroIdx = 4*n1 + predIdx
                Y[i][nonZeroIdx] = 1
                
            else:
                print("Fatal error: label size must be either 2 or 16")
                return
        
        if shuffle_b > 0.5:
#                print "I shuffle ..."
#            print v2
            if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
            elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                shuffleFlanks(v = X[i], flankSize = flankSizeOut, shuffleLength = shuffleLength, inner_b = inner_b)
            else:
                shuffle(X[i])          
        
        #add reversed and complemented sequence: 
        if augmentWithRevComplementary_b == 1:
            
            print("Fatal error: augmentWithRevComplementary_b is not in working order here!")
            return 
        
            
    if convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        

     
#    if transformStyle_b == 0:  
         
    return X, Y


#ONLY IMPLEMENTED FOR labelsCodetype = 0
def genSamplesDirectlyFromGenome(genomeString,
                                 nrSamples,
                                 augmentWithRevComplementary_b = 0,
                                 flankSize = 50, 
                                 labelsCodetype = 0,
                                 outputAsString_b = 1,
                                 outputEncodedOneHot_b = 0,
                                 outputEncodedInt_b = 0,
               outputEncodedType = 'int8',
               getOnlyRepeats_b = 0,
               repeatArray = '',
               shuffle_b = 0,
               shuffleLength = 0,
               inner_b = 0):
    
    
    '''
    Input:
       genomeString: genome sequnece as a string (as output by readGenome) 
       nrSamples: nr of samples to generate
       augmentWithRevComplementary_b: if 1, generate samples for reverse complementary string too
       flankSize: length of flank (left/right); ie 50 means a total of 100 with 50 on each side of the mid to be predicted
       labelsCodetype: just set to 0
       outputAsString_b: if 1 output as string 
       outputEncodedOneHot_b: if 1 output as one-hot encoded
       outputEncodedInt_b: if 1 output as int-encoded 
       outputEncodedType: int-type to be used when asking for encoded 
       getOnlyRepeats_b: if 1 fetch only smaples at repeat positions
       repeatArray: repeat info sequnce (as output by readGenome)
       shuffle_b: if 1 shuffle shuffleLength of the flanks (see inner_b too)
       shuffleLength: see shuffle_b
       inner_b: if 1 shuffle the inner part, if 0 the outer part of the flanks.
           
           Only one of outputAsString_b, outputEncodedOneHot_b, outputEncodedInt_b should be set to 1.
    
    Output: tuple X, Y
    
    X: the list of string-samples or numpy-array of encoded samples
    Y: the corresponding labels (letter or encoded-letter)
    
    '''
    
    
    #Set a labels-shape depending on the labelsCodetype:
    if labelsCodetype == 0:
        
        labelShape = 4
    
    elif labelsCodetype == 1 or labelsCodetype == -1 or labelsCodetype == 3:
        
        labelShape = 2 
        
    elif labelsCodetype == 2:
        
        labelShape = 3
    
    if outputAsString_b == 1:
        
        X = []
        Y = []
        Xrev = []
        Yrev = []
        
    elif outputEncodedOneHot_b == 1:

        try:
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            
    elif outputEncodedInt_b == 1:

        try:
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
#                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
        
      
    if outputAsString_b == 1:
    
        lGenome = len(genomeString)
        print("length of genome string:", lGenome)
    
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site 
            idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
    
            #If only repeat-positions are wanted:
            if getOnlyRepeats_b == 1:
                
                while repeatArray[idx] == 0:
                    
                    idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
                
            if labelsCodetype == 0:
                Y.append(genomeString[idx])
    #        elif labelsCodetype == 1:
    #            Y[i] = basePair(genomeArray[idx])
    #        elif labelsCodetype == -1:
    #            Y[i] = basePairType(genomeArray[idx])
    #        elif labelsCodetype == 2:
    #            if exonicArray[idx] == 1:
    #                Y[i] = exonicInd
    #            elif repeatArray[idx] == 1:
    #                Y[i] = repeatInd
    #            else:
    #                Y[i] = otherInd
    #        elif labelsCodetype == 3: #repeat or not?
    #            if repeatArray[idx] == 1:
    #                Y[i] = repeatBinInd
    #            else:
    #                Y[i] = notRepeatBinInd
                
    #            print "idx:", idx
                        
                
            #... and fetch the correspondning flanks:
            X.append(genomeString[(idx - flankSize):idx] + genomeString[(idx+1):(idx + 1 + flankSize)]) #left-hand flank + right-hand flank
            
               
            
            if shuffle_b > 0.5:
    #                print "I shuffle ..."
    #            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])     
                    
                
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
            
                xRev = X[i][::-1]
                xRevAug = ''.join(map(complementLetter, xRev)) #join the letters .. to form a string 
                Xrev.append(xRevAug)
                if labelsCodetype == 0:
                    Yrev.append(complementLetter(Y[i]))
    #            elif labelsCodetype == 1:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == -1:
    #                Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
    #            elif labelsCodetype == 2:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == 3:
    #                Y[i + nrSamples ] = Y[i]
    
                if shuffle_b > 0.5:
    #                print "I shuffle ..."
    #            print v2
                    if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                        shuffleFlanks(v = Xrev[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                    elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                        shuffleFlanks(v = Xrev[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                    else:
                        shuffle(Xrev[i]) 
                    
               
    #    if transformStyle_b == 0: 
        
        if augmentWithRevComplementary_b == 1:
            
            X = X + Xrev
            Y = Y + Yrev
    


    elif outputEncodedOneHot_b == 1:
    
        lGenome = len(genomeString)
        print("length of genome string:", lGenome)
    
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site 
            idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
    
            #If only repeat-positions are wanted:
            if getOnlyRepeats_b == 1:
                
                while repeatArray[idx] == 0:
                    
                    idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
                
            if labelsCodetype == 0:
                Y[i] = oneHotLetter(genomeString[idx])
    #        elif labelsCodetype == 1:
    #            Y[i] = basePair(genomeArray[idx])
    #        elif labelsCodetype == -1:
    #            Y[i] = basePairType(genomeArray[idx])
    #        elif labelsCodetype == 2:
    #            if exonicArray[idx] == 1:
    #                Y[i] = exonicInd
    #            elif repeatArray[idx] == 1:
    #                Y[i] = repeatInd
    #            else:
    #                Y[i] = otherInd
    #        elif labelsCodetype == 3: #repeat or not?
    #            if repeatArray[idx] == 1:
    #                Y[i] = repeatBinInd
    #            else:
    #                Y[i] = notRepeatBinInd
                
    #            print "idx:", idx
                        
                
            #... and fetch the correspondning flanks:
            X[i][:flankSize] = np.asarray(list(map(oneHotLetter, genomeString[(idx - flankSize):idx]))) #left-hand flank
            X[i][flankSize:] = np.asarray(list(map(oneHotLetter, genomeString[(idx+1):(idx + 1 + flankSize)]))) #right-hand flank
            
               
            
            if shuffle_b > 0.5:
    #                print "I shuffle ..."
    #            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])     
                    
                
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
            
                xRev = X[i][::-1]
                xRevAug = np.asarray(list(map(complementArray, xRev)))
                X[i + nrSamples] = xRevAug
                if labelsCodetype == 0:
                    Y[i + nrSamples ] = complementArray(Y[i])
    #            elif labelsCodetype == 1:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == -1:
    #                Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
    #            elif labelsCodetype == 2:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == 3:
    #                Y[i + nrSamples ] = Y[i]
                    
               
    #    if transformStyle_b == 0:  


         
    
    elif outputEncodedInt_b == 1:
        
    
        lGenome = len(genomeString)
        print("length of genome string:", lGenome)
    
        #sample from genomeSeq:
        i = 0
        for i in range(nrSamples):
            
            #Get random site 
            idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
    
            #If only repeat-positions are wanted:
            if getOnlyRepeats_b == 1:
                
                while repeatArray[idx] == 0:
                    
                    idx = np.random.randint(flankSize+1,lGenome-flankSize) 
    
                
            if labelsCodetype == 0:
                Y[i] = intLetter(genomeString[idx])
    #        elif labelsCodetype == 1:
    #            Y[i] = basePair(genomeArray[idx])
    #        elif labelsCodetype == -1:
    #            Y[i] = basePairType(genomeArray[idx])
    #        elif labelsCodetype == 2:
    #            if exonicArray[idx] == 1:
    #                Y[i] = exonicInd
    #            elif repeatArray[idx] == 1:
    #                Y[i] = repeatInd
    #            else:
    #                Y[i] = otherInd
    #        elif labelsCodetype == 3: #repeat or not?
    #            if repeatArray[idx] == 1:
    #                Y[i] = repeatBinInd
    #            else:
    #                Y[i] = notRepeatBinInd
                
    #            print "idx:", idx
                        
                
            #... and fetch the correspondning flanks:
            X[i][:flankSize] = np.asarray(list(map(intLetter, genomeString[(idx - flankSize):idx]))) #left-hand flank
            X[i][flankSize:] = np.asarray(list(map(intLetter, genomeString[(idx+1):(idx + 1 + flankSize)]))) #right-hand flank
            
               
            
            if shuffle_b > 0.5:
    #                print "I shuffle ..."
    #            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i])     
                    
                
            #add reversed and complemented sequence: 
            if augmentWithRevComplementary_b == 1:
            
                xRev = X[i][::-1]
                xRevAug = np.asarray(list(map(complementInt, xRev)))
                X[i + nrSamples] = xRevAug
                if labelsCodetype == 0:
                    Y[i + nrSamples ] = complementInt(Y[i])
    #            elif labelsCodetype == 1:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == -1:
    #                Y[i + nrSamples ] = complementArrayBasepairType(Y[i])
    #            elif labelsCodetype == 2:
    #                Y[i + nrSamples ] = Y[i]
    #            elif labelsCodetype == 3:
    #                Y[i + nrSamples ] = Y[i]
                    
               
    #    if transformStyle_b == 0:  
        
        
    else:
        
        print("Use one-hot or int-encoding or none (string output)")

    return X, Y  


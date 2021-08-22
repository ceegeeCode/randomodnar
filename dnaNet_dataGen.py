#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:01:36 2019

@author: Christian Grønbæk (except fastReadGenome) Desmond Elliott (fastReadGenome)

"""

'''
The code below -- all in a 'research state' -- was used for managing data input for training/testing the neural networks 
reported in the following paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 


Notes:

    -- all code is in a 'research state'. Don't expect perfect doc-strings or great usage tutorials. But there are
        some examples and explanation below.
        
    -- The code below contains much that was not used for the paper
    -- For the paper we used the following functions: 
    * splitGenomeInChromosomes
    * fastReadGenome
    * encodeGenome
    * genSamplesForDynamicSampling_I (training)
    * getAllSamplesFromGenome (testing/predictions)
    * the small utils at the beginning of the coding section (up to "Utils for restructuring data")
    * functions for various chekcs: checkChromoSeqs, checkOneHotEncoding, checkArrays
    -- other function were used in earlier experiments (readGenome, genSamplesDirectlyFromGenome)    



The main functions right now are:

splitGenomeInChromosomes
fastReadGenome, readGenome
encodeGenome
genSamplesForDynamicSampling_I
getAllSamplesFromGenome 

For input/output see their doc-strings (or, better, find the function's code below). What they do:    

splitGenomeInChromosomes: read in a complete assembly string for a genome (.fa) and splits it in files per chromosome.
These individual chromosome files are used in the predictions.

fastReadGenome/readGenome: reads in a .fa formatted file. The human genome assembly hg19.fa, for instance. This
can be downloaded via http://hgdownload.cse.ucsc.edu/downloads.html#human (it's explained there how, it's
just a wget really).    
    
encodeGenome: reads in putput from readGenome and spits out an encoded versions; encodings: one-hot or
as integers (0,1,2,3). At the very top of the code you'll find how the four letters (A C G T) 
are encoded (A, C, G, T corr's to 0,1,2,3 etc)  
    
genSamplesForDynamicSampling_I: is the sampling fct used for the dynamical sampling (e.g in the fct
allInOneWithDynSampling_ConvLSTMmodel found in dnaNet_LSTM)

getAllSamplesFromGenome: gets all "samples" of flanks and corr midpoints moving across a genome. Important: 
this will get the encoded bases at all positions, also those containing a not-ACGT letter! So to use the 
output, handling the non-qualified letters (encoded by the array for the wild card, W) must be done in the 
application. Used when: obtaining the prediction accross the genome (dnaNet_stats module)

In early experiments (for language models -- the transformer) we used also the function
genSamplesDirectlyFromGenome: generates samples from a genome string (read in by readGenome)  



---------------------------------------------------------------------------------------------

Special note on the functions splitGenomeInChromosomes and fastReadGenome:

These are improved version of the functions of the same name with an "_v1" appended.
The latter versions were used for the "Predict DNA ... " paper. The version "_v1"
have an uninteded behaviour: they only reset the read-in when the length of the chromosome 
-- lines starting with '>' --- are below a set length. Thus the last passed chromosome
having a name of length below the threshold will keep being read in until a new 
chromosome with that property appears in the file; all chromosomes --- or patches or ... -- 
in between but of longer names than the threshold will be appended to the last 
chromosome of shorter name last passed. In the new version (which is without "_v1")
this is fixed.

The implications of this bad code (_v1 version): 
A. The single autosomal chromosome seq's (used in predictions and all analysis downstream from that) 
are !not! hit since they were read in with the chromoNameBound set high to have no effect (1000); 
only for mouse the value was set to have an effect (65), where the autosomals chr's were not hit
(only chr Y is hit) 
B. for training the chromosomeNameBound was generally set low (100, 65 for mouse) but the setting
actually had no effect -- the full genome file was simply read in (up to the boundary length set); 
thus "extra" pieces of sequence in the assembly could be read in in addition to the sequence parts of 
the chromosomes. While unintentional this should not do any harm, since it is all valid sequence. 
For human and yeast this did not happen (only full chromosome sequences were in the
genome files); for zebrafish there was also no effect since only the first 2 billion bases were read in
and these all belonged to full chromosomal seqeunces. For mouse the training used the first 3 billion 
bases, which happened to consist of all the chromosomal sequences and (maybe) an additional about 5 million 
bases worth of shorter sequences were read in, or a part of it.

More details, difference means diff between results of using the new version and the _v1 version: 
1) human hg38: no differences (pre-filtered the assembly)
2) mouse m38: only diff's in Y chromo; can have a small effect on training (not really 
erroneous, only an extra about 4-5 million bases appended); for predictions only autosomals 
were used 
3) yeast: no diff's
4) for drosophila: _v1 added bases to the last chromo read in (X); no diff's implied since
only used first 150 million was used for training and only autosomals for prediction
(and the single chrom seq's were read in with chromoNameBound = 1000)
5) for zebrafish: no diff's for training since first 2 billion bases used (and these
sit in autosomals read in first); for single chromo seq's no diff's found 
(were read in with chromoNameBound = 1000) so no implications for predictions
6) For mouse, human and yeast the individual chromosome sequences obtained by the _v1 
versions were checked against direct downloads of the chromo seq's from UCSC and 
only very few diff's were seen (no diffs in mouse, yeast; less than 50 in each human chromo
and typically less if any; table in Suppl Data check

---------------------------------------------------------------------------------------------


##################################################################################################
# Usage:
##################################################################################################

The calls/examples can be used in a python console (e.g with Spyder or a Jupyter notebook) by copying the part you 
want to run(just ctrl-c the selected lines) and then pasting them at the python-prompt in the console (just ctrl-v 
there). And then press shift+enter or whatever key strokes it takes for executing the commands in the python console.

In the remaining part of this usage section, right below, you'll a find a lot of calls to some of the functions 
in this dataGen module. And many variations on setting the actual arguments' for some of the function calls (e.g
the are many calls to splitGenomeInChromosomes and checkArrays, and you may first see a long list of various possible 
argument values, all for handling the genome files of various organisms). Many of the main functions are though called 
only in other functions (e.g. fastReadGenome, genSamplesForDynamicSampling_I).


#Example

#Read in chr22 of human ref genome, assembly hg38
import dnaNet_dataGen as dataGen

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"
fileName = r"hg38_chr22.txt"
fileGenome = rootGenome +fileName


Xall, X, Xrepeat, Xexonic = dataGen.fastReadGenome(fileGenome, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 10, 
               startAtPosition = 0,
               endAtPosition = 1e10, #some gigantic number
               outputAsDict_b = 0,
#               outputGenomeString_b = 0,
               randomChromo_b = 0, 
               onlyTheseChromos_b = 0,
               onlyTheseChromos = [],
               avoidChromo = [])




####################################################

Import module:

####################################################

import dnaNet_dataGen as dataGen

#################################################
## One-off's: to get genome seq split over chromo's
#################################################

import dnaNet_dataGen as dataGen

#Human:

#assembly hg38, Yuhu's download:

#genome seq of all chr's: /isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/GCF_000001405.38_GRCh38.p12_genomic_filter.fna
import dnaNet_dataGen as dataGen

organism = 'human'
chromoNameBound = 1000
rootOut = '/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
dataGen.splitGenomeInChromosomes(root = '/isdata/kroghgrp/wzx205/scratch/01.SNP/00.Data/', genomeFileName = 'GCF_000001405.38_GRCh38.p12_genomic_filter.fna', genomeShortName = 'hg38', rootOut = rootOut, organism = organism, chromoNameBound = chromoNameBound)


#assembly hg38, own download:
import dnaNet_dataGen as dataGen
chromoNameBound = 10
rootGenome = '/isdata/kroghgrp/tkj375/data/DNA/human/hg38/'
genomeFileName = 'hg38.fa'
genomeShortName = 'hg38'
rootOut = '/isdata/kroghgrp/tkj375/data/DNA/human/hg38/'
dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = genomeFileName, genomeShortName = genomeShortName, rootOut = rootOut, chromoNameBound = chromoNameBound)


#T2T chm13 chr8.
import dnaNet_dataGen as dataGen
chromoNameBound = 10
rootGenome = '/isdata/kroghgrp/tkj375/data/DNA/human/T2T_chm13/'
genomeFileName = 'chm13.chr8_v9.fasta'
genomeShortName = 't2t_chm13'
rootOut = rootGenome
dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = genomeFileName, genomeShortName = genomeShortName, rootOut = rootOut, chromoNameBound = chromoNameBound)


#Mouse:
import dnaNet_dataGen as dataGen

#m38
organism = 'mouse'
chromoNameBound = 1000 #65
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
genomeFileName =  r"Mus_musculus.GRCm38.dna_sm.primary_assembly.fa"
genomeShortName = 'm38'
rootOut = '/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/newSplitFct/'
lengthInfo = dataGen.splitGenomeInChromosomes_v1(root = rootGenome, genomeFileName = genomeFileName, genomeShortName = genomeShortName, rootOut = rootOut, chromoNameBound = chromoNameBound, organism = 'mouse')



#mm9
organism = 'mouse'
chromoNameBound = 65
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/"
genomeFileName =  r"mm9.fa"
genomeShortName = 'mm9'
rootOut = '/isdata/kroghgrp/tkj375/data/DNA/mouse/mm9/'
dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = genomeFileName, genomeShortName = genomeShortName, rootOut = rootOut, chromoNameBound = chromoNameBound, organism = 'mouse')



#Yeast:

import dnaNet_dataGen as dataGen
rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
fileName = r"S288C_reference_sequence_R64-1-1_20110203.fsa"

organism = 'yeast'
genomeShortName = 'R64' 
rootOut = rootGenome

chromoNameBound = 1000
dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = fileName, genomeShortName = genomeShortName, rootOut = rootOut, organism = organism, chromoNameBound = chromoNameBound)


#Drosophila:

import dnaNet_dataGen as dataGen

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/drosophila/"
fileName = r"dmel-all-chromosome-r6.18.fasta"

organism = 'drosophila'
genomeShortName = 'r6.18' 
rootOut = rootGenome + r'newSplitFct/'

chromoNameBound =1000
lengthInfo = dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = fileName, genomeShortName = genomeShortName, rootOut = rootOut, organism = organism, chromoNameBound = chromoNameBound)

#Zebrafish:

import dnaNet_dataGen as dataGen

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/zebrafish/GRCz11/ncbi-genomes-2020-01-05/"
fileName = r"GCF_000002035.6_GRCz11_genomic.fna"

organism = 'zebrafish'
genomeShortName = 'GRCz11'
rootOut = rootGenome 
chromoNameBound = 1000
dataGen.splitGenomeInChromosomes(root = rootGenome, genomeFileName = fileName, genomeShortName = genomeShortName, rootOut = rootOut, organism = organism, chromoNameBound = chromoNameBound)



#################################################
## Checks
#################################################


import dnaNet_dataGen as dataGen

#human
root1 =r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/'
chromoList1 =  ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
#chromoList1 =  ['hg38_chr20']
ext1 = '.txt'

root2 = r'/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/checkChromoSeqs/'
chromoList2 =  ['chr22', 'chr21', 'chr20', 'chr19', 'chr18', 'chr17', 'chr16', 'chr15', 'chr14', 'chr13', 'chr12', 'chr11', 'chr10', 'chr9', 'chr8', 'chr7', 'chr6', 'chr5', 'chr4', 'chr3', 'chr2', 'chr1']
#chromoList2 =  ['chr20']
ext2 = '.fa'


#mouse, m38:
root1 =r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
chromoList1  = [ 'm38_chr1', 'm38_chr2',   'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6',  'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11', 'm38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
#chromoList1 =  ['m38_chr16']
ext1 = '.txt'

root2 = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/checkChromoSeqs/"
chromoList2 =  ['chr1', 'chr2', 'chr3',  'chr4', 'chr5',  'chr6',  'chr7', 'chr8', 'chr9', 'chr10', 'chr11',  'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19']
#chromoList2 =  ['chr16']
ext2 = '.fa'



#yeast:
root1 =r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/"
chromoList1  = ['R64_chr1', 'R64_chr2', 'R64_chr3', 'R64_chr4', 'R64_chr5', 'R64_chr6', 'R64_chr7', 'R64_chr8','R64_chr9', 'R64_chr10', 'R64_chr11', 'R64_chr12','R64_chr13', 'R64_chr14', 'R64_chr15', 'R64_chr16']
#chromoList1 =  ['m38_chr16']
ext1 = '.txt'

root2 = r"/isdata/kroghgrp/tkj375/data/DNA/yeast/R64/S288C_reference_genome_R64-1-1_20110203/checkChromoSeqs/"
chromoList2 =  ['chrI', 'chrII',  'chrIII',  'chrIV',  'chrV', 'chrVI', 'chrVII', 'chrVIII', 'chrIX', 'chrX', 'chrXI', 'chrXII', 'chrXIII', 'chrXIV', 'chrXV', 'chrXVI'] # 'chrM']
#chromoList2 =  ['chr16']
ext2 = '.fa'

#Check that the chromosome seq's we have used (by splitting the full genome seq) are id to the strings that can
#be had by downloading the genome-files for the indicvidual chromosomes:
useFastReadGenome_b = 1
resultsDict = dataGen.checkChromoSeqs(root1 = root1, fileList1 = chromoList1, ext1 = ext1, root2 = root2, fileList2= chromoList2, ext2 = ext2, useFastReadGenome_b = useFastReadGenome_b)
print resultsDict


for chr in resultsDict:
    print chr, resultsDict[chr]

#dump result:
import cPickle as pickle
#dumpFile = root2 + 'resultsDict_checkChromoSeqs.p'
dumpFile = root2 + 'resultsDict_checkChromoSeqs_useFastReadGenome.p'
pickle.dump(resultsDict, open(dumpFile, "wb"))

dumpFile2 = root2 + 'resultsDict_useFastReadGenome.p'
resultsDict2 = pickle.load(open(dumpFile2, "rb"))

#to check chromo lengths vs those posted by UCSC: download the UCSC chromo-length file, read it in, convert to dict and do the check: 
import pandas as pd
genomePrefix = 'm38_'
lengthsFile = root1 + r'/ucsc/mm10.chrom.sizes'
chromoLengthsUCSC = pd.read_csv(lengthsFile, sep = '\t', header=None)
#chromoLengthsUCSC.columns = ['chromo', 'length']
ucscLengthlist = chromoLengthsUCSC.values.tolist()
ucscLengthDict  = {}
for i in range(len(ucscLengthlist)):
    chr = genomePrefix  + ucscLengthlist[i][0]
    length = ucscLengthlist[i][1]
    ucscLengthDict[chr] = length
#check the lengths:
diffs = 0
for chromo in resultsDict:
    length = resultsDict[chromo][1]
    lengthUCSC = ucscLengthDict[chromo]
    if length != lengthUCSC:
        diffs +=1
    print length, lengthUCSC, diffs


    
#Check that our encoding inverts to the input sequnce:
resultsDict = dataGen.checkOneHotEncoding(rootGenome = root1, chromosomeList = chromoList1)
import cPickle as pickle
dumpFile = root1 + 'resultsDict_checkOneHotEncoding.p'
pickle.dump(resultsDict, open(dumpFile, "wb"))


    

######################################################
# Check label and GC arrays
######################################################

import dnaNet_dataGen as dataGen

##########################
#For human, hg38
##########################

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/GRCh38.p12/"

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeDict = {'hg38_chr22':[10500000,1e9], 'hg38_chr21':[5010000,1e9], 'hg38_chr20':[0,1e9], 'hg38_chr19':[0,1e9], 'hg38_chr18':[0,1e9], 'hg38_chr17':[0,1e9], 'hg38_chr16':[0,1e9], 'hg38_chr15':[17000000,1e9], 'hg38_chr14':[16000000,1e9], 'hg38_chr13':[16000000,1e9], 'hg38_chr12':[0,1e9], 'hg38_chr11':[0,1e9], 'hg38_chr10':[0,1e9], 'hg38_chr9':[0,1e9], 'hg38_chr8':[0,1e9], 'hg38_chr7':[0,1e9], 'hg38_chr6':[0,1e9], 'hg38_chr5':[0,1e9], 'hg38_chr4':[0,1e9], 'hg38_chr3':[0,1e9], 'hg38_chr2':[0,1e9], 'hg38_chr1':[0,1e9]}


(If we run checkArrays on #LSTM1, the flankSize will ot match what was actually used for the GCcontent arrays (viz flanks 50), so we'll see some
(wrong) discrepancies in the GCcontent; but we could run it on LSTM1 to see that the labelArrays for LSTM1 are fine.
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/"
modelFileNameNN ="modelLSTM_augWithCompl_2Conv2LayerLstm_flanks200_win3_filters64and256_stride1_overlap0_dropout00_dense50_bigLoopIter0_repeatNr96"
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM1/notAvgRevCompl/"
flankSize = 200
)


#LSTM4/5, flanks 50, trained on hg38; GC bias only uses the q-arrays, and we use the flanks 50 version then and not the 200: 
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/LSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr176'
flankSize = 50


#GC bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/human/on_hg38/GCbias/notAvgRevCompl/"
forATorGCbias_b = 1 #!!!!!!!!!!!!

chromosomeOrderList = ['hg38_chr20']
random_b = 0 #checking all seg's in one named chromo

chromosomeOrderList = ['hg38_chr22', 'hg38_chr21', 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16', 'hg38_chr15', 'hg38_chr14', 'hg38_chr13', 'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']
chromosomeOrderList = [ 'hg38_chr20', 'hg38_chr19', 'hg38_chr18', 'hg38_chr17', 'hg38_chr16',  'hg38_chr12', 'hg38_chr11', 'hg38_chr10', 'hg38_chr9', 'hg38_chr8', 'hg38_chr7', 'hg38_chr6', 'hg38_chr5', 'hg38_chr4', 'hg38_chr3', 'hg38_chr2', 'hg38_chr1']

random_b = 1 #check about 10 pct of l segments in all chromos

##########################
#For mouse, GRCm38:
##########################

rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/mouse/GRCm38/"
chromosomeDict = { 'm38_chr1':[0,1e9],'m38_chr2':[0,1e9],'m38_chr3':[0,1e9], 'm38_chr4':[0,1e9], 'm38_chr5':[0,1e9], 'm38_chr6':[0,1e9],'m38_chr7':[0,1e9],'m38_chr8':[0,1e9],'m38_chr9':[0,1e9],'m38_chr10':[0,1e9],'m38_chr11':[0,1e9],'m38_chr12':[0,1e9],'m38_chr13':[0,1e9],'m38_chr14':[0,1e9],'m38_chr15':[0,1e9],'m38_chr16':[0,1e9],'m38_chr17':[0,1e9],'m38_chr18':[0,1e9],'m38_chr19':[0,1e9]}

#Mouse model (same settings as the human LSTM4) used for predicting on the mouse m38 genome:
rootOutput = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/notAvgRevCompl/"
rootModel = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/mouseLSTM4/trainTestSplit_80_20/"
modelFileNameNN = 'modelLSTM_1LayerConv2LayerLstm1LayerDense50_flanks50_win4_filters256_stride1_overlap0_dropout00_bigLoopIter0_repeatNr193'
flankSize = 50

#GC bias:
modelFileName_forATorGCbias = 'GCbias'
rootOutput_forATorGCbias = r"/isdata/kroghgrp/tkj375/various_python/DNA_proj/results_nets/ptPrecious/mouse/on_GRCm38/GCbias/notAvgRevCompl/"
forATorGCbias_b = 1 #!!!!!!!!!!!!


random_b = 0 #checking all seg's in one named chromo
chromosomeOrderList = [ 'm38_chr19']


chromosomeOrderList = [ 'm38_chr1', 'm38_chr2', 'm38_chr3', 'm38_chr4',  'm38_chr5', 'm38_chr6', 'm38_chr7', 'm38_chr8', 'm38_chr9', 'm38_chr10', 'm38_chr11','m38_chr12', 'm38_chr13', 'm38_chr14', 'm38_chr15', 'm38_chr16', 'm38_chr17', 'm38_chr18', 'm38_chr19']
random_b = 1 #check about 10 pct of l segments in all chromos



##########################
#General settings:


segmentLength = 1000000

augmentWithRevComplementary_b = 0  #!!!!!

#window lgth and stepsize used in generating the avg prediction
windowLength = 1
stepSize = 1
                      
outDict = dataGen.checkArrays(rootGenome = rootGenome, 
                         chromosomeDict = chromosomeDict,
                        chromosomeOrderList = chromosomeOrderList,  
                            rootOutput = rootOutput,
                            rootModel = rootModel,
                            flankSize = flankSize,
                             modelFileName = modelFileNameNN,  
                             segmentLength = segmentLength,
                             averageRevComplementary_b = augmentWithRevComplementary_b,
                             windowLength = windowLength,
                             stepSize = stepSize,
                             forATorGCbias_b = forATorGCbias_b, 
                             rootOutput_forATorGCbias= rootOutput_forATorGCbias,
                             modelFileName_forATorGCbias = modelFileName_forATorGCbias,
                             random_b = random_b)

#Results: 

#Human, hg38:
For chr 19 and 20 full: no diffs. Across chromos, random 10 pct seg's of each, no diffs:
>>> for k in outDict:
...     res = 0
...     for s in outDict[k]:
...             res += np.asarray(outDict[k][s])
...     print k, res
...
hg38_chr4 [8000000       0       0       0]
hg38_chr9 [12289344        0        0        0]
hg38_chr19 [6939950       0       0       0]
hg38_chr18 [7944092       0       0       0]
hg38_chr17 [14947838        0        0        0]
hg38_chr16 [10273070        0        0        0]
hg38_chr5 [14000000        0        0        0]
hg38_chr8 [14949900        0        0        0]
hg38_chr3 [22989950        0        0        0]
hg38_chr12 [16941652        0        0        0]
hg38_chr11 [9999714       0       0       0]
hg38_chr10 [13815613        0        0        0]
hg38_chr7 [13000000        0        0        0]
hg38_chr2 [18999767        0        0        0]
hg38_chr1 [27956391        0        0        0]
hg38_chr20 [5000000       0       0       0]
hg38_chr6 [19829484        0        0        0]

#Mouse, GRCm38:
For chr 19 full: no diffs. Across chromos, random 10 pct seg's of each, no diffs:
>>> for k in outDict:
...     res = 0
...     for s in outDict[k]:
...             res += np.asarray(outDict[k][s])
...     print k, res
...
m38_chr18 [12000000        0        0        0]
m38_chr19 [2873990       0       0       0]
m38_chr12 [11999950        0        0        0]
m38_chr13 [14979854        0        0        0]
m38_chr10 [17924019        0        0        0]
m38_chr11 [8000000       0       0       0]
m38_chr16 [6000000       0       0       0]
m38_chr17 [9000000       0       0       0]
m38_chr14 [11999842        0        0        0]
m38_chr15 [7960562       0       0       0]
m38_chr8 [12956506        0        0        0]
m38_chr9 [9000000       0       0       0]
m38_chr1 [16907040        0        0        0]
m38_chr2 [17000000        0        0        0]
m38_chr3 [19799583        0        0        0]
m38_chr4 [11826294        0        0        0]
m38_chr5 [8999183       0       0       0]
m38_chr6 [18000000        0        0        0]
m38_chr7 [12920976        0        0        0]



'''

import numpy as np
import sys


from random import shuffle

import cPickle as pickle

import os

import re


###################################################################
## Defs and Utils
###################################################################

#codeA = np.array([1,0,0,0])
#codeG = np.array([0,1,0,0])
#codeC = np.array([0,0,1,0])
#codeT = np.array([0,0,0,1])

VERBOSE = False  # used to control really verbose print statements

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

codeW = [3,3,3,3] #an array diff form the one-hots for ACGT and easy to recognize


codeA_asArray = np.asarray(codeA, dtype ='int8')
codeC_asArray = np.asarray(codeC, dtype ='int8')
codeG_asArray = np.asarray(codeG, dtype ='int8')
codeT_asArray = np.asarray(codeT, dtype ='int8')

codeW_asArray = np.asarray(codeW, dtype ='int8')

#codeA_asArray = np.asarray(codeA, dtype ='float32')
#codeC_asArray = np.asarray(codeC, dtype ='float32')
#codeG_asArray = np.asarray(codeG, dtype ='float32')
#codeT_asArray = np.asarray(codeT, dtype ='float32')


codeA_asInt = 0
codeC_asInt = 1
codeG_asInt = 2
codeT_asInt = 3

codeW_asInt = 7

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
    else:
        return codeW_asArray


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
    else: 
        return 'N'


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
    else:
        return codeW_asInt

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
    else:
        return 'W'
    
    
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
    else:
        return 7

def complementLetter(x):
    
    if x == 'A':
        return 'T'
    if x == 'T':
        return 'A'
    if x == 'C':
        return 'G'
    if x == 'G':
        return 'C'
    if x == 'W':
        return 'W'
        
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
    else:
        return codeW_asArray


def complementArray(x):
    
    if np.array_equal(x, codeA_asArray):
        return codeT_asArray
    elif np.array_equal(x, codeT_asArray):
        return codeA_asArray
    elif np.array_equal(x, codeC_asArray):
        return codeG_asArray
    elif np.array_equal(x, codeG_asArray):
        return codeC_asArray
    else:
        return codeW_asArray
        

def complementInt(x):
    
    if np.array_equal(x, codeA_asInt):
        return codeT_asInt
    elif np.array_equal(x, codeT_asInt):
        return codeA_asInt
    elif np.array_equal(x, codeC_asInt):
        return codeG_asInt
    elif np.array_equal(x, codeG_asInt):
        return codeC_asInt
    else:
       return codeW_asInt

        

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
## Utils for restructuring data
###################################################################
    

def splitGenomeInChromosomes(root, genomeFileName, genomeShortName, rootOut, chromoNameBound = 10, organism = 'human'):
    
    '''As first part of readGenome/fastReadGenome, but spits out the strings for each chromo
    to a file. Files are named: [genomeShortName]_chr*.txt, with * = chr number.'''
    
        
    pathToGenomeFile =root +  genomeFileName
    handle = open(pathToGenomeFile)
    accumulatedLength = 0
    lineCnt = 0
    currNr = -1
    currChromo = ''
    currChromoFile  = ''
#    Xchr = ''
    lengthDict = {}
    lenChr = 0
    posInChr = 0
    
    defaultChrNr = 0
    
    #gray mouse lemur file has particular structure
    if organism == 'grayMouseLemur':
        pattern = r'(>NC[\S]+)( Microcebus murinus isolate mixed chromosome )([0-9]+)(, Mmur_3.0, whole genome shotgun sequence)'
        compPattern = re.compile(pattern)

    # Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length
    lengthChromoOK_b = 0
    while True:
        lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
        if not lines:
            break  # no more lines to read

        for line in lines:
        
            v = line.strip()

            if lineCnt == 0:
                print "Genome data file 1st line:\n ", line
                

            if '>' in v:
                
                defaultChrNr += 1
                
                if currChromo != '' and lengthChromoOK_b == 1:
                    currChromoFile.close()
                    
                    lengthDict[genomeShortName + '_' + 'chr' + currNr] = lenChr  
                    print "Length chromo %s recorded: %d" % (genomeShortName + '_' + 'chr' + currNr, lenChr) 
                
                print v
                # only check the length of line when necessary
                if len(v[1:]) >= chromoNameBound: 
                    
                    lengthChromoOK_b = 0
                
                elif len(v[1:]) < chromoNameBound:  
                    
                    lengthChromoOK_b = 1
                                           
                    currChromo = v[1:]
                    if organism == 'human' or organism == 'mouse':
                       
                        if v[1:4] == 'chr':
                            currNr = v[4:]
                        elif v[2] != ' ':
                            currNr = v[1:3]
                        else:
                            currNr = v[1]
                            
                    elif organism == 'drosophila':
                        
#                        if v[2] != ' ':
#                            currNr = v[1:3]
#                        else:
#                            currNr = v[1]
                            
                        if v[1:3] in ['2R','2L', '3R', '3L'] and v[3] == ' ':
                            currNr = v[1:3]
                        elif v[1] in ['4', 'X', 'Y'] and v[2] == ' ':
                            currNr = v[1]
                        else:
                            currNr = v[1:4]
                    
                   
                    elif organism == 'yeast':
                         
                       print v
                       currNr = str(defaultChrNr)
                       
                    elif organism == 'grayMouseLemur':
                        
                        vMatch = compPattern.match(v)
                        
                        if vMatch:
                            
                            currNr = vMatch.group(3)
                            print "gray mouse lemur; currNr by re-match: ", currNr
                            
                        else:
                            
                            print "S..... er ......"
                            lengthChromoOK_b = 0 #slight abuse ...
                            continue
                            
                        
                        
#                        if v[59] == ',':
#                            
#                            currNr = v[57:59]
#                            
#                        elif v[58] == ','
#                        
#                            currNr = v[57]                      
#                        else:
#                            
#                            print "No current chromo nr defined!"
#                            continue
                  
                    else:
                        
                        currNr = str(defaultChrNr)
                        
                    
                    currChromoFilePath = rootOut + genomeShortName + '_' + 'chr' + currNr + '.txt'
                    print "currChromoFilePath ", currChromoFilePath
                    #flush file if exists:
                    if os.path.exists(currChromoFilePath):
                        currChromoFile = open(currChromoFilePath, 'w')
                        currChromoFile.close()
                       
                    currChromoFile = open(currChromoFilePath, 'a')
                    currChromoFile.write(line)
#                    chromoList.append([currChromo, lenChr])    
                    print("Found data for this chromosome: %s. Will be named %s" % (currChromo, genomeShortName + '_' + 'chr' + currNr))
                    #Xchr = ''
                    lenChr = 0
                    posInChr = 0
    
                     
            else:
                
                if lengthChromoOK_b ==1:
 
                    thisLength = len(v)
                    lenChr += thisLength
                    accumulatedLength += thisLength
                    posInChr += thisLength
    
                    #Xchr += v 
                    currChromoFile.write(line)
    #                Xall += v
    #                lenXall += len(v)
    #            
                    
            lineCnt += 1 

    handle.close()

    #record for lastly passed chromo too:
    lengthDict[currChromo] = lenChr     
            
    print(lengthDict)        
#        print("Length of genome sequence read in:%d" % lenXall)
    
    #dump it
    dumpFile = rootOut + 'chromosomeLengthDict'
    pickle.dump(lengthDict, open(dumpFile, "wb") )
    
    return lengthDict



#this is the version of splitGenomes ... use dfor the "Predict DNA .." paper; the version above is better; see notes
#in the preamble!
def splitGenomeInChromosomes_v1(root, genomeFileName, genomeShortName, rootOut, chromoNameBound = 10, organism = 'human'):
    
    '''As first part of readGenome/fastReadGenome, but spits out the strings for each chromo
    to a file. Files are named: [genomeShortName]_chr*.txt, with * = chr number.'''
    
    pathToGenomeFile =root +  genomeFileName
    handle = open(pathToGenomeFile)
    accumulatedLength = 0
    lineCnt = 0
    currChromo = ''
    currChromoFile  = ''
#    Xchr = ''
    chromoList = []
    lenChr = 0
    posInChr = 0
    
    defaultChrNr = 0
    

    # Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length
    while True:
        lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
        if not lines:
            break  # no more lines to read

        for line in lines:
        
            v = line.strip()

            if lineCnt == 0:
                print "Genome data file 1st line:\n ", line
                
            
            if '>' in v:
                
                defaultChrNr += 1
                
                print v
                # only check the length of line when necessary
                if len(v[1:]) < chromoNameBound:  
                
                    if currChromo != '':
                        currChromoFile.close()
                        
                        chromoList.append([currChromo, lenChr])  
                    
                       
                    currChromo = v[1:]
                    if organism == 'human' or organism == 'drosophila' or organism == 'mouse':
                       
                        if v[1:4] == 'chr':
                            currNr = v[4:]
                        elif v[2] != ' ':
                            currNr = v[1:3]
                        else:
                            currNr = v[1]
                   
                    elif organism == 'yeast':
                        
                       print v
                       currNr = str(defaultChrNr)
                  
                    else:
                        
                        currNr = str(defaultChrNr)
                        
                    
                    currChromoFilePath = rootOut + genomeShortName + '_' + 'chr' + currNr + '.txt'
                    print "currChromoFilePath ", currChromoFilePath
                    #flush file if exists:
                    if os.path.exists(currChromoFilePath):
                        currChromoFile = open(currChromoFilePath, 'w')
                        currChromoFile.close()
                       
                    currChromoFile = open(currChromoFilePath, 'a')
                    currChromoFile.write(line)
#                    chromoList.append([currChromo, lenChr])    
                    print("Found data for this chromosome: %s" % currChromo)
                    #Xchr = ''
                    lenChr = 0
                    posInChr = 0
                                     
            else:
                thisLength = len(v)
                lenChr += thisLength
                accumulatedLength += thisLength
                posInChr += thisLength

                #Xchr += v 
                currChromoFile.write(line)
#                Xall += v
#                lenXall += len(v)
#            
                    
            lineCnt += 1 

    handle.close()

    #record for lastly passed chromo too:
    #chromoList.append([currChromo, lenChr])    
            
    print(chromoList)        
#        print("Length of genome sequence read in:%d" % lenXall)
    
    
    return chromoList



def generateRepeatArraysGenomeSeqMasking(rootGenome, chromosomeList, rootOutput, saveRepeatArray_b = 1):
    '''Get a boolean array indicating the repeat masked part in a genome string (eg soft masked from UCSC
    genome browser). This functions also finds the length of each chromosome and stores this in a dict.'''  
    

    #Loop over chr's; call fastReadGenome and get length of the chromo; dump results.    

    lenghtDict  = {}
        
    for chromoName in chromosomeList:
        
        fileName = chromoName + ".txt"
        fileGenome = rootGenome + fileName
        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition = 0
        endAtPosition  = 3e9
        outputAsDict_b = 0
        outputGenomeString_b = 1 #!!!
        randomChromo_b = 0
        avoidChromo = []


        Xall, X, Xrepeat, Xexonic = fastReadGenome(fileName = fileGenome, 
                   exonicInfoBinaryFileName = exonicInfoBinaryFileName,
                   chromoNameBound = chromoNameBound, 
                   startAtPosition = startAtPosition,
                   endAtPosition = endAtPosition,
                   outputAsDict_b = outputAsDict_b,
#                   outputGenomeString_b = outputGenomeString_b,
                   randomChromo_b = randomChromo_b, 
                   avoidChromo = avoidChromo)
                   
        lenghtDict[chromoName] = len(X)
        
        if saveRepeatArray_b == 1:
            repeatArray = np.asarray(Xrepeat, dtype = 'int8')
            #dump it
            dumpFile = rootOutput + chromoName + '_annotationArray_repeatsGenomeSeq'
            pickle.dump(repeatArray, open(dumpFile, "wb") )
            
            if repeatArray.shape[0] != lenghtDict[chromoName]:
                
                print "Warning: length of genomic seq of chromo %s is %d so diff from length of repeats array (%d)!" % (chromoName, lenghtDict[chromoName], repeatArray.shape[0])
 
    
    #dump it
    dumpFile = rootOutput + 'chromosomeLengthDict'
    pickle.dump(lenghtDict, open(dumpFile, "wb") )



def checkChromoSeqs(root1, fileList1, ext1, root2, fileList2, ext2, useFastReadGenome_b = 1):
    '''Compares list of pairs of files, fasta, txt ..., containing strings; comparison
    letter-by-letter. The fileLists should match.
    
    Purpose: to check that the chromo-file we obtain by splitting a downloaded
    genome-seq is id to the one that can be downloaded for that chromosome (from 
    same assembly/genome-site of course). Obs: we only check that the letters are
    id, not their case (: we convert all to uppercase); the case of the letters is
    used only for repeat masking, and disregarding that is indeed what we want here.
    
    useFastReadGenome_b: if set (1), the DNA string is read in using the fastReadGenome fct (that fct reads
    in the string as the readGenome fct). The settings in the call to fastReadGenome is as in 
    predictOnChromosomes (in dnaNet_stats module), since this is what we really want to be sure of: that
    the string we use for predictions is as it should be!
    
    ''' 
    
    resultsDict = {} #to hold final output

    #loop over the chromo-list:
    nrOfChromos = len(fileList1)
    if nrOfChromos == len(fileList2):
        print("nr of chromos to  check: %d", nrOfChromos)
    else:
        print("Lists do no contain the same number of files. So: exit.")
        return
    
    notPassedList = []
    passedList = []
    
    roots = [root1, root2]
    files = [fileList1, fileList2]
    exts = [ext1, ext2]
    

    for i in range(nrOfChromos):
        
        chromo = files[0][i]
        
        Xlist = [] #to contain the two strings
    
        #step1: read in the files line by line; concatenate all lines after the first '>'.
        #step2: check that the two strings are of equal length
        #step3: loop through the strings and count number of diffs
        
        for j in range(2):
            
            #step1:
            pathToGenomeFile = roots[j] + files[j][i] + exts[j]
            print("Looking up data in file: ", pathToGenomeFile )
            handle = open(pathToGenomeFile)
            accumulatedLength = 0
            lineCnt = 0
            lenChr = 0
            Xchr = '' #to contain the string for current file

            if useFastReadGenome_b != 0:
            
                #Read in data from genome; 
                #use settings as in predictOnChromosomes (in dnaNet_stats module):
                exonicInfoBinaryFileName = ''
                chromoNameBound = 100
                startAtPosition, endAtPosition  = 0, int(1e9)
                outputAsDict_b = 0
                randomChromo_b = 0
                avoidChromo = []
            
                Xall, X, Xrepeat, Xexonic = fastReadGenome_v1(fileName = pathToGenomeFile, 
                       exonicInfoBinaryFileName = exonicInfoBinaryFileName,
                       chromoNameBound = chromoNameBound, 
                       startAtPosition = startAtPosition,
                       endAtPosition = endAtPosition,
                       outputAsDict_b = outputAsDict_b,
        #               outputGenomeString_b = outputGenomeString_b,
                       randomChromo_b = randomChromo_b, 
                       avoidChromo = avoidChromo)
                       
                Xall = Xall.upper()                       
                Xlist.append(Xall)
               
            else: #reading in as in (fast)readGenome, but not calling that function
        
                while True:
                    lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
                    if not lines:
                        break  # no more lines to read
            
                    for line in lines:
                    
                        v = line.strip()
            
                        if lineCnt == 0:
                            print "Genome data file 1st line:\n ", line
                        
                        #skip lines with a '>'; these header lines will preceed the dna-seq
                        #or the dna-seq for each chromosome 
            #            print v
                        if '>' in v:
                            
                            print v
                            
                            currChromo = v[1:]
                            print("Found data for this chromosome: %s" % currChromo)
                            Xchr = ''
                            lenChr = 0
                            posInChr = 0
                                             
                        else:
                            thisLength = len(v)
                            lenChr += thisLength
                            accumulatedLength += thisLength
                            posInChr += thisLength
            
                            Xchr += v.upper() 
                                
                        lineCnt += 1 
                        
                handle.close()
                Xlist.append(Xchr)
            
        #Do the check, step2/3:
        X1 = Xlist[0]
        L1 = len(X1)
        print("X1 has length %d", L1 )
        X2 = Xlist[1]
        L2 = len(X2)
        print("X2 has length %d", L2 )
        if L1 != L2:
            
            print("The two files contain diff number of letters! Not passed ...")
            notPassedList.append(chromo)
            resultsDict[chromo] = 'na', L1, L2

        else:

            print("The two files contain the same number of letters! Checking further --- ")

            nrOfIdLetters = 0  
            nrOfDiffLetters = 0
            for k in range(L1):
                
                if X1[k] == X2[k]:
                    nrOfIdLetters +=1
                else:
                    nrOfDiffLetters +=1
                    
            if nrOfIdLetters == L1:
                
                print("Chromo %s Passed!", chromo)
                passedList.append(chromo)
                
            else:
                print("Nr of diff %d letters in chromo %s", nrOfDiffLetters, chromo )
                
            resultsDict[chromo] = nrOfDiffLetters, L1
        
        
        if useFastReadGenome_b == 0:
            dumpFile = root1 + 'resultsDict_checkChromoSeqs.p'
        else:
            dumpFile = root1 + 'resultsDict_checkChromoSeqs_useFastReadGenome.p'
    
        pickle.dump(resultsDict, open(dumpFile, "wb"))
        print("Dumped results for chromosome %s", chromo)        
            
    if passedList == fileList1:
        
        print("All chromos passed!")
                
    return resultsDict
    

def checkOneHotEncoding(rootGenome, chromosomeList):  
    '''Check of our one-hot encoding for the listed chromosoems by first encoding 
    and then decoding (back to original) and then comparing that string letter-by-letter 
    to the input string.'''
    
    #Step1: read-in the genome string from file (string1); this uses the (intentionally) same
    #procedure as used in (fast)ReadGenome. In a separate check we test that this
    #string has the 'official' length, ie the one stated at the download site
    
    #Step2: encode the string from file using the encodeGenome fct; this implicitly
    #uses (fast)ReadGenome for reading the file. The fct call also returns the genome
    #string that is encoded (string2). 
    
    #Step3: check that string2 is id to string1.
    #Step4: convert the encoded string back to 4-letters (and wildcard N); out: string3
    #step5: check that string3 is id to string1.
    
    notPassed = []
    resultsDict = {}
    nChecked = 0
    for chromoName in chromosomeList:
        
        fileName = chromoName + ".txt"
        fileGenome = rootGenome +fileName
        print("Looking up data in file: ", fileGenome )
        handle = open(fileGenome)
        accumulatedLength = 0
        lineCnt = 0
        lenChr = 0
        Xchr = '' #to contain the string for current file

        while True:
            lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
            if not lines:
                break  # no more lines to read
    
            for line in lines:
            
                v = line.strip()
    
                if lineCnt == 0:
                    print "Genome data file 1st line:\n ", line
                
                #skip lines with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
    #            print v
                if '>' in v:
                    
                    print v
                    
                    currChromo = v[1:]
                    print("Found data for this chromosome: %s" % currChromo)
                    Xchr = ''
                    lenChr = 0
                    posInChr = 0
                                     
                else:
                    thisLength = len(v)
                    lenChr += thisLength
                    accumulatedLength += thisLength
                    posInChr += thisLength
    
                    Xchr += v.upper() 
                        
                lineCnt += 1 
            
        handle.close()
    
        #step2
        print("Doing the encoding ...")
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition, endAtPosition  = 0, int(1e9) #this covers our use
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b = 1 #!!!
        randomChromo_b = 0
        avoidChromo = []
    
        #encoding the data:
        encodedGenomeData =  encodeGenome(fileName = fileGenome, 
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
                               
        print("Encoding done.")
                               
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString =  encodedGenomeData 
        
        #step3
        L1 = len(Xchr)
        print("String1 has length %d", L1 )
        L2 = len(genomeSeqString)
        print("String2 (DNA string to be encoded) has length %d", L2 )
        L3 = genomeSeq.shape[0]
        print("String3 (encoded string) has length %d", L3 )
        if L1 != L2:
            
            print("The two files contain diff number of letters! Check of this chromo %s is skipped!", chromoName)
            notPassed.append(chromoName)
            resultsDict[chromoName] = L1, L2, L3, 'na'
            continue            

        else: #is string1 == string2?
            
            nrOfIdLetters = 0  
            nrOfDiffLetters = 0
            for k in range(L1):
                
                if genomeSeqString[k] == 'W' and Xchr[k] == 'N':
                    nrOfIdLetters +=1
                elif genomeSeqString[k] == Xchr[k].upper():
                    nrOfIdLetters +=1
                else:
                    nrOfDiffLetters +=1
                    
        if nrOfIdLetters == L1:
            
            print("For chromo %s all letters in string1 and string2 are id ", chromoName)
            
        else:
            print("For chromo %s, nr of diff letters in string1 and string2: %d", chromoName, nrOfDiffLetters )
                
        #step4: decode the encoded string:
        if L3 != L1:
            print("The encoded string has length diff from string2! Check of this chromo %s is skipped!", chromoName)
            notPassed.append(chromoName)
            resultsDict[chromoName] = L1, L2, L3, 'na'
            continue 

#        inversion = map(invOneHotLetter, genomeSeq)
        inversion = np.apply_along_axis(invOneHotLetter, 1, genomeSeq)
            
        nrOfIdLetters2 = 0  
        nrOfDiffLetters2 = 0            
        for k in range(L3):
            
#            letter = invOneHotLetter(genomeSeq[k])
            letter = inversion[k]
                
            if letter == Xchr[k].upper():
                nrOfIdLetters2 +=1
            else:
                nrOfDiffLetters2 +=1
                    
        if nrOfIdLetters == L1:
            
            print("For chromo %s all letters in string1 and string3 are id ", chromoName)
            
        else:
            print("For chromo %s, nr of diff letters in string1 and string3: %d", chromoName, nrOfDiffLetters2 )
        
        resultsDict[chromoName] = L1, L2, L3, nrOfDiffLetters, nrOfDiffLetters2
        
        dumpFile = rootGenome + 'resultsDict_checkOneHotEncoding.p'
        pickle.dump(resultsDict, open(dumpFile, "wb"))
        
        nChecked +=1 
        
    print("Nr of chromos checked %d", nChecked)
    print("Nr of chromos not passed %d", len(notPassed))
    if len(notPassed) > 0:
        print("The not-passede chomos: %s", notPassed)
        
    return resultsDict
    
    


def checkArrays(rootGenome, 
                         chromosomeDict,
                         chromosomeOrderList, 
                         rootOutput,
                         rootModel,
                         modelFileName,
                         flankSize,
                         extension = ".txt",
                        segmentLength = 1e6,
                        averageRevComplementary_b = 0, #!!!!!
                        startAtSegmentDict = {}, 
                        windowLength = 1,
                        stepSize = 1,
                        forATorGCbias_b = 0,
                        rootOutput_forATorGCbias = '',
                        modelFileName_forATorGCbias = '',                             
                          random_b = 1): 
    '''
    
    chromosomeOrderList: just gives the order in which the function processes the 
    chromosomes. The list should only contain chromo names that are keys in chromosomeDict.
    '''
    
    outDict= {}
    
    for chromoName in chromosomeOrderList:
                
        fileName = chromoName + extension
        fileGenome = rootGenome + fileName
        
        #Read in data from genome and get it encoded:
        exonicInfoBinaryFileName = ''
        chromoNameBound = 100
        startAtPosition, endAtPosition  = chromosomeDict[chromoName]
        outputEncoded_b = 1
        outputEncodedOneHot_b = 1
        outputEncodedInt_b = 0
        outputAsDict_b = 0
        outputGenomeString_b = 1 #!!!
        randomChromo_b = 0
        avoidChromo = []
    
        #encoding the data:
        encodedGenomeData =  encodeGenome(fileName = fileGenome, 
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
        
        
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqString =  encodedGenomeData 

    
        lGenomeSeq = len(genomeSeq)
        
        nrSegments = int(np.floor(float(lGenomeSeq)/segmentLength))   
        
        print "nrSegments: ", nrSegments
        
        rootOutput_thisChromo =  rootOutput + chromoName + r'/'
        rootOutput_forATorGCbias_thisChromo = rootOutput_forATorGCbias + chromoName + r'/'

        
        #Read-in labelArray and GC-content array:
        outDict[chromoName] = {}
        for i in range(nrSegments): #range(nrSegments):

            #if desired (default) we only check about 10 pct of the segments:
            go_b = 1
            if random_b == 1:
                
                go_b = np.random.binomial(1, p= 0.1)
                
            if go_b == 0:
                continue
        
            genomeIdNameSeg = genomeIdName + '_segment' + str(i)
            
            try:
                loadFile = rootOutput_thisChromo + modelFileName + '_' + 'qualifiedArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)            
                print loadFile
                qualArraySeg = pickle.load(open( loadFile, "rb"))
                
                loadFile = rootOutput_thisChromo + modelFileName + '_' + 'labelArray' + '_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b)
                print loadFile
                labelArraySeg = pickle.load(open( loadFile, "rb"))
                
                inversion = np.apply_along_axis(invOneHotLetter, 1, labelArraySeg)            
                
                loadFile = rootOutput_forATorGCbias_thisChromo +  modelFileName_forATorGCbias + '_predReturn_' + genomeIdNameSeg + '_avgRevCompl' + str(averageRevComplementary_b) + '_win' + str(windowLength) + '_step' + str(stepSize)   
                print loadFile
                gcArraySeg, cntCorr_seg, cntTot_seg, cntCorrRep_seg, cntTotRep_seg, args_seg = pickle.load(open( loadFile, "rb"))
                
                            
            except IOError:
                continue
            
            cntQual = 0
            cntDiffLabel = 0
            cntDiffGC = 0       
            cntDiffAT = 0
            
            if i == 0:
                segStartAt = max(flankSize, startAtPosition)
            else:
                segStartAt = i*segmentLength + startAtPosition
            
            effSegmentLength = len(qualArraySeg)
                
            for j in range(effSegmentLength):
                
                if qualArraySeg[j] == 0:
                    continue
                
                cntQual +=1 
                
                pos = segStartAt + j
                
                #Is 
                if inversion[j] != genomeSeqString[pos].upper():
                    cntDiffLabel += 1
                
                if genomeSeqString[pos].upper() == 'G' or genomeSeqString[pos].upper() == 'C':
                    
                    if gcArraySeg[j] != 1:
                        
                        cntDiffGC += 1
                        
                elif genomeSeqString[pos].upper() == 'A' or genomeSeqString[pos].upper() == 'T':
                    
                    if gcArraySeg[j] != 0:
                        
                        cntDiffAT += 1
                        
            print "For chromo ", chromoName, "the qual/diff were ", cntQual, cntDiffLabel, cntDiffGC, cntDiffAT 
                        
            outDict[chromoName][i] = [cntQual, cntDiffLabel, cntDiffGC, cntDiffAT]
            
    return outDict



#Gather the info about input:
def gatherGenomeInfo(rootGenome, chromosomeOrderList, chromosomeLengthDict, chromosomeCheckDict, chromosomeOneHotDict):
    '''Make table giving per chromo: length, nr of diff's to downloadable sequence (fct: checkChromoSeqs in 
    dnaNet_dataGen module), no of diff's in check of 1-hot encoding (fct: checkOneHotEncoding in dnaNet_dataGen 
    module).
    '''

#    #fecth chromo-lgth dict:
#    loadFile = rootGenome + r'/chromosomeLengthDict'
#    chromosomeLengthDict = pickle.load(open( loadFile, "rb"))
#    
#    #fetch chromoCheck dict:
#    loadFile = rootGenome + r'/chromosomeLengthDict'
#    chromosomeCheckDict = pickle.load(open( loadFile, "rb"))
#
#    #fetch oneHotCheckDict = {}
#    loadFile = rootGenome + r'/chromosomeLengthDict'
#    chromosomeOneHotDict = pickle.load(open( loadFile, "rb"))
    
    
    chromoInfoDict = {}
    for chromo in chromosomeOrderList:
        
        L = chromosomeLengthDict[chromo]
        
        nrOfDiffLetters, L1 = chromosomeCheckDict[chromo]
        L1_, L2, L3, nrOfDiffLetters_, nrOfDiffLetters2 = chromosomeOneHotDict[chromo]
        
        if L1 != L: 
            print "Obs: chromo lght is %d but checked lgth is %d" % (L, L1)
        if L1_ != L: 
            print "Obs: chromo lght is %d but encoded lgth is %d" % (L, L1_)
        
        chromoInfoDict[chromo] = {'length':L, 'diffsChrCheck':nrOfDiffLetters, 'diffsOnHotCheck':nrOfDiffLetters_}
        
    return chromoInfoDict

        
        

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
#               outputGenomeString_b =0, not needed: we always output the genome string
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
        cntNotACGT = 0
        for i in range(lenXall):
                            
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
            else:
                X += 'W'
                Xrepeat.append(0)
                Xexonic.append(XexonicAll[i])
                cntNotACGT +=1
                if VERBOSE:   
                    print("Letter is: %s so not ACGTacgt at: %d" % (Xall[i],i))
                
                
        #If desired the letters will be "one-hot" encoded:
        lenX = len(X)
        print("Length genome sequence, only ACGT's:%d; not ACGT's: %d" % (lenX - cntNotACGT, cntNotACGT))
        
        
        
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
            print("Warning: lengths of exonic info and dna-seq differ!")

        
         
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
            cntNotACGT = 0
            for i in range(lenXchrAll):
                
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
                    Xchr += 'W' 
                    XchrRepeat.append(0)
                    XchrExonic.append(XchrExonicAllDict[chromo][i])
                    cntNotACGT +=1
                    
                    #Only if N's can only be heading or trailing does this makes good sense for the railing ones; the heading-N count is ok:
                    if XchrAllDict[chromo][i] == 'N':   
                        if trailing_b == 1:
                            trailingNs += 1
                        else:
                            headingNs += 1
                        
                    if VERBOSE:   
                        print("Letter is: %s so not ACGTacgt at: %d" % (XchrAllDict[chromo][i],i))
                
                    
                    
            XchrDict[chromo] = Xchr
            XchrRepeatDict[chromo] = XchrRepeat
            XchrExonicDict[chromo] = XchrExonic
            XchrNsDict[chromo] = [headingNs, trailingNs]
            
            lenXchr = len(XchrDict[chromo])  
            print("Length genome sequence, for chromo %s: only ACGT's:%d; not ACGT's: %d" % (chromo, lenXchr - cntNotACGT, cntNotACGT))
 

        return XchrAllDict, XchrDict, XchrRepeatDict, XchrExonicDict, XchrNsDict


def fastReadGenome(fileName, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 1000, 
               startAtPosition = 0,
               endAtPosition = int(1e26), #some gigantic number
               outputAsDict_b = 0,
#               outputGenomeString_b =0, not needed: we always output the genome string
               randomChromo_b = 0, 
               onlyTheseChromos_b = 0,
               onlyTheseChromos = [],
               avoidChromo = []):
    
    '''
    TODO: Optimise the implementation when outputAsDict_b = 1
    TODO: implement effectiveStartAtPosition, effectiveEndAtPosition when outputAsDict_b = 1

    This rewrite combines two previously separate steps if outputAsDict_b == 0.

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

    startAtPosition = int(startAtPosition)
    endAtPosition = int(endAtPosition)
    
    XexonicChromo = '' #to contain the full sequence of all 0/1 exonic indicators for (a desired part of) a chromo
    Xall = '' #to contain the full sequence of all letters in the read-in genome
    XexonicAll = '' #to contain the full sequence of all 0/1 exonic indicators in the read-in genome
    X = '' #to only contain the ACGT's of the genome
    Xrepeat = [] #to contain the indicator of a position being in a repeat or not
    Xexonic = [] #to contain the indicator of a position being exonic or not
    Xchr = '' #to only contain the ACGT's in one chromosome of the read-in genome
    XchrExonic = '' #to only contain the exonic-info in one chromosome of the read-in genome
#    XchrRepeat = [] #to contain the indicator of a position being in a repeat or not
#    XchrExonic = [] #to contain the indicator of a position being exonic or not
#    XchrAllDict = {} #dict to map: each chromosome to its sequence
#    XchrExonicAllDict = {} #dict to map: each chromosome to its exonic-info sequence
#    XchrDict = {} #dict to map: each chromosome to list [its length, sequence(only ACGT's)]
#    XchrRepeatDict = {} #repeat info corr to XchrDict
#    XchrExonicDict = {} #exonic info corr to XchrDict
#    XchrNsDict = {} #to collect the number of heading,trailing ends for each chromo
    lineCnt = 0
    lenXall = 0
    lenX = 0
#    lenXchrAll = 0
#    lenXchr = 0
    
    print("Fast reading in genome data ... ")
    print("Only considering data following fasta header lines (: chromo names \n for eucaryots) of length < %d" %  chromoNameBound)



    lenChr = 0
    posInChr = 0
    chromoList = []
    exonicInfoList = []
    currChromo = ''
    handle = open(fileName)
    accumulatedLength = 0

    if exonicInfoBinaryFileName == '':
        # This operation does not need to be repeated. If the file does not exist when the script is called, it will not exist during the reading of the file
        #if no exon-info file was provided, we use 0:
        print("OBS: no file containing exonic info was provided, so exonic status is set to 0 from {} - {}".format(lenXall, endAtPosition+1))
        XexonicChromo += str(0) * (endAtPosition + 1 - lenXall)

    passedStartPosition_b = 0
    if outputAsDict_b == 0:
        # Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length
        lengthChromoOK_b = 0 
        while True:
            lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
            if not lines:
                break  # no more lines to read

            for line in lines:
            
                v = line.strip()

                if lineCnt == 0:
                    print("Genome data file 1st line:\n ", line)
                
                #skip lines with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
                if '>' in v:
                    
                    
                    #check if we want to avoid data for this chromosome:
                    if len(avoidChromo) > 0:
                        
                        skip_b = 0
                        cnt = 0
                        for chromoName in avoidChromo:
                            pattern = re.compile(chromoName)
                            isThere = re.search(pattern, v)
                            if isThere: 
                                print("isThere? %s " % isThere.group())
                                #abuse this boolean rater than introducing one more (to save time in if below)
                                lengthChromoOK_b = 0 
                                print("Skip data for this chromosome: %s" % v)
                                skip_b = 1
                                avoidChromo.pop(cnt)
                                break
                            cnt +=1

                        if skip_b == 1:
                            continue
                    
                    if len(v[1:]) >= chromoNameBound:   
                    
                        lengthChromoOK_b = 0
                    # only check the length of line when necessary
                    elif len(v[1:]) < chromoNameBound:  
                        
                        lengthChromoOK_b = 1
                    
                        if currChromo != '':
                            chromoList.append([currChromo, lenChr])    
                    
                        currChromo = v[1:]
                        print("Found data for this chromosome: %s" % currChromo)
                        lenChr = 0
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
                            
                            exonicInfoList.append([currChromo, len(XexonicChromo)])                            
                else:
                    
                    if lengthChromoOK_b == 1:
                        thisLength = len(v)
                        lenChr += thisLength
                        accumulatedLength += thisLength
                        posInChr += thisLength
                        if accumulatedLength >= startAtPosition:
    
                            XexonicPieceToAdd = XexonicChromo[(posInChr - thisLength):posInChr]
                            #record first passage of startAtPosition to determine where the read-in seq actually starts
                            if passedStartPosition_b == 0:
                                #accumulatedLength - thisLength #index of first entry in present v
                                #startAtPosition is found at the  startAtPosition - (accumulatedLength - thisLength) +1'th position of v.
                                #As v is indexed from 0 we then take this tail of v:
                                v = v[(startAtPosition - (accumulatedLength - thisLength)):] #index of first entry in present v
                                print len(v), accumulatedLength, thisLength  
                                XexonicPieceToAdd = XexonicPieceToAdd[(startAtPosition - (accumulatedLength - thisLength)):]
                                passedStartPosition_b = 1
                                
                            if accumulatedLength <= endAtPosition:
                                Xall += v
                                XexonicAll += XexonicPieceToAdd
                        #            print "Xall at %d: %s" % (lineCnt, Xall)
                        #            if lineCnt ==2:
                        #                return 
                                lenXall += len(v)
                            else:
                                #capture the potential final piece up to endAtPosition (indexing considerations as above):
                                v= v[:(endAtPosition - (accumulatedLength - thisLength) + 1)]
                                XexonicPieceToAdd = XexonicPieceToAdd[:(endAtPosition - (accumulatedLength - thisLength) + 1)]
                                Xall += v
                                XexonicAll += XexonicPieceToAdd
                                break
                        
                lineCnt += 1 
                
#        effectiveEndAtPosition = effectiveStartAtPosition + lenXall -1

        handle.close()

        #record for lastly passed chromo too:
        chromoList.append([currChromo, lenChr])    
                
        print(chromoList)
        print(exonicInfoList)
        
        print("Length of genome sequence read in:%d" % lenXall)
        print("Length of exonic-info sequence read in:%d" % len(XexonicAll))
        if lenXall != len(XexonicAll):
            print("Warning: lengths of exonic info and dna-seq differ!")
        
        #not all letters are ACGT!:
        zeroSet = {'A', 'T', 'C', 'G'}
        oneSet  = {'a', 't', 'c', 'g'}
        for i in range(lenXall):
            if i % 100000000 == 0:
                print("ACGTacgt checked {} tokens".format(i))
            if Xall[i] in zeroSet:
                X += Xall[i]
                Xrepeat.append(0)
                Xexonic.append(XexonicAll[i])
            elif Xall[i] in oneSet:
                X += Xall[i].upper()
                Xrepeat.append(1)
                Xexonic.append(XexonicAll[i])
            else:
                X += 'W'
                Xrepeat.append(0)
                Xexonic.append(XexonicAll[i])
                # It isn't an IndexError so we shouldn't call one
                if VERBOSE:
                	print("Letter %s not ACGTacgt at: %d" % (Xall[i],i))

        #If desired the letters will be "one-hot" encoded:
        lenX = len(X)
        print("Length genome sequence, ACGT's and W's:%d" % lenX)
        print("Of these %d are W's" % X.count('W'))
        
        return Xall, X, Xrepeat, Xexonic #, effectiveStartAtPosition, effectiveEndAtPosition

    else: #outputAsDict_b != 0:  
        
        # TODO: Optimise this if we ever want to use it. It has some expensive operations
        #If desired we only read in data from one randomly chosen chromosome
        
        print "The outputAsDict_b != 0 option is not implemented"
        
 


#Version of fastReadGenome used in "Predict DNA ..."
def fastReadGenome_v1(fileName, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 1000, 
               startAtPosition = 0,
               endAtPosition = int(1e26), #some gigantic number
               outputAsDict_b = 0,
#               outputGenomeString_b =0, not needed: we always output the genome string
               randomChromo_b = 0, 
               onlyTheseChromos_b = 0,
               onlyTheseChromos = [],
               avoidChromo = []):
    
    '''
    TODO: Optimise the implementation when outputAsDict_b = 1
    TODO: implement effectiveStartAtPosition, effectiveEndAtPosition when outputAsDict_b = 1

    This rewrite combines two previously separate steps if outputAsDict_b == 0.

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

    startAtPosition = int(startAtPosition)
    endAtPosition = int(endAtPosition)
    
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
#    XchrNsDict = {} #to collect the number of heading,trailing ends for each chromo
    lineCnt = 0
    lenXall = 0
    lenX = 0
    lenXchrAll = 0
    lenXchr = 0
    
    print("Fast reading in genome data ... ")
    print("Only considering data following fasta header lines (: chromo names \n for eucaryots) of length < %d" %  chromoNameBound)



    lenChr = 0
    posInChr = 0
    chromoList = []
    exonicInfoList = []
    currChromo = ''
    handle = open(fileName)
    accumulatedLength = 0

    if exonicInfoBinaryFileName == '':
        # This operation does not need to be repeated. If the file does not exist when the script is called, it will not exist during the reading of the file
        #if no exon-info file was provided, we use 0:
        print("OBS: no file containing exonic info was provided, so exonic status is set to 0 from {} - {}".format(lenXall, endAtPosition+1))
        XexonicChromo += str(0) * (endAtPosition + 1 - lenXall)

    passedStartPosition_b = 0
    if outputAsDict_b == 0:
        # Loop through the file to find the diff chromo's, their lengths and check if the exonic-info seq's match in length
        while True:
            lines = handle.readlines(100000)  # read in chunks of 100,000 lines http://effbot.org/zone/readline-performance.htm 
            if not lines:
                break  # no more lines to read

            for line in lines:
            
                v = line.strip()

                if lineCnt == 0:
                    print("Genome data file 1st line:\n ", line)
                
                #skip lines with a '>'; these header lines will preceed the dna-seq
                #or the dna-seq for each chromosome 
                if '>' in v:
                    # only check the length of line when necessary
                    if len(v[1:]) < chromoNameBound:  
                    
                        if currChromo != '':
                            chromoList.append([currChromo, lenChr])    
                    
                        currChromo = v[1:]
                        print("Found data for this chromosome: %s" % currChromo)
                        lenChr = 0
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
                            
                            exonicInfoList.append([currChromo, len(XexonicChromo)])                            
                else:
                    thisLength = len(v)
                    lenChr += thisLength
                    accumulatedLength += thisLength
                    posInChr += thisLength
                    if accumulatedLength >= startAtPosition:

                        XexonicPieceToAdd = XexonicChromo[(posInChr - thisLength):posInChr]
                        #record first passage of startAtPosition to determine where the read-in seq actually starts
                        if passedStartPosition_b == 0:
                            #accumulatedLength - thisLength #index of first entry in present v
                            #startAtPosition is found at the  startAtPosition - (accumulatedLength - thisLength) +1'th position of v.
                            #As v is indexed from 0 we then take this tail of v:
                            v = v[(startAtPosition - (accumulatedLength - thisLength)):] #index of first entry in present v
                            print len(v), accumulatedLength, thisLength  
                            XexonicPieceToAdd = XexonicPieceToAdd[(startAtPosition - (accumulatedLength - thisLength)):]
                            passedStartPosition_b = 1
                            
                        if accumulatedLength <= endAtPosition:
                            Xall += v
                            XexonicAll += XexonicPieceToAdd
                    #            print "Xall at %d: %s" % (lineCnt, Xall)
                    #            if lineCnt ==2:
                    #                return 
                            lenXall += len(v)
                        else:
                            #capture the potential final piece up to endAtPosition (indexing considerations as above):
                            v= v[:(endAtPosition - (accumulatedLength - thisLength) + 1)]
                            XexonicPieceToAdd = XexonicPieceToAdd[:(endAtPosition - (accumulatedLength - thisLength) + 1)]
                            Xall += v
                            XexonicAll += XexonicPieceToAdd
                            break
                        
                lineCnt += 1 
                
#        effectiveEndAtPosition = effectiveStartAtPosition + lenXall -1

        handle.close()

        #record for lastly passed chromo too:
        chromoList.append([currChromo, lenChr])    
                
        print(chromoList)
        print(exonicInfoList)
        
        print("Length of genome sequence read in:%d" % lenXall)
        print("Length of exonic-info sequence read in:%d" % len(XexonicAll))
        if lenXall != len(XexonicAll):
            print("Warning: lengths of exonic info and dna-seq differ!")
        
        #not all letters are ACGT!:
        zeroSet = {'A', 'T', 'C', 'G'}
        oneSet  = {'a', 't', 'c', 'g'}
        for i in range(lenXall):
            if i % 100000000 == 0:
                print("ACGTacgt checked {} tokens".format(i))
            if Xall[i] in zeroSet:
                X += Xall[i]
                Xrepeat.append(0)
                Xexonic.append(XexonicAll[i])
            elif Xall[i] in oneSet:
                X += Xall[i].upper()
                Xrepeat.append(1)
                Xexonic.append(XexonicAll[i])
            else:
                X += 'W'
                Xrepeat.append(0)
                Xexonic.append(XexonicAll[i])
                # It isn't an IndexError so we shouldn't call one
                if VERBOSE:
                	print("Letter %s not ACGTacgt at: %d" % (Xall[i],i))

        #If desired the letters will be "one-hot" encoded:
        lenX = len(X)
        print("Length genome sequence, ACGT's and W's:%d" % lenX)
        print("Of these %d are W's" % X.count('W'))
        
        return Xall, X, Xrepeat, Xexonic #, effectiveStartAtPosition, effectiveEndAtPosition

    else: #outputAsDict_b != 0:  
        # TODO: Optimise this if we ever want to use it. It has some expensive operations
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
        zeroSet = {'A', 'T', 'C', 'G'}
        oneSet  = {'a', 't', 'c', 'g'}
        for chromo in XchrAllDict.keys():
            
            print("Now at chromosome: %s" % chromo)
            
            Xchr = ''
            XchrRepeat = []
            XchrExonic = []
            
            lenXchrAll = len(XchrAllDict[chromo])
            
            print("Length of this genome sequence:%d" % lenXchrAll)
            
            cntNotACGT = 0
            for i in range(lenXchrAll):
                
                if i % 100000000 == 0:
                    print("ACGTacgt checked {} tokens".format(i))
                    
                if XchrAllDict[chromo][i] in zeroSet:            
                    Xchr += XchrAllDict[chromo][i]
                    XchrRepeat.append(0)
                    XchrExonic.append(XchrExonicAllDict[chromo][i])
                elif XchrAllDict[chromo][i] in oneSet:            
                    Xchr += XchrAllDict[chromo][i].upper()
                    XchrRepeat.append(1)
                    XchrExonic.append(XchrExonicAllDict[chromo][i])
                else:
                    Xchr += 'W' 
                    XchrRepeat.append(0)
                    XchrExonic.append(XchrExonicAllDict[chromo][i])
                    cntNotACGT +=1

                                
            XchrDict[chromo] = Xchr
            XchrRepeatDict[chromo] = XchrRepeat
            XchrExonicDict[chromo] = XchrExonic
            
            lenXchr = len(XchrDict[chromo])  
            print("Length genome sequence, for chromo %s: only ACGT's:%d; not ACGT's: %d" % (chromo, lenXchr - cntNotACGT, cntNotACGT))
 

        return XchrAllDict, XchrDict, XchrRepeatDict,  {} #the empty dixt is just placeholder for the heading-trailing N info dict (which we do not care about here, for the sake of cutting down processing time) 



def encodeGenome(fileName, 
               exonicInfoBinaryFileName = '',
               chromoNameBound = 1000, 
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
    defined by codeA, codeT, codeC and codeG. Not-ACGT's are encoded as wild-card (W), codeW.

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
    Not-ACGT's in the genome string are encoded as wild-cards, and by the array
    
    codeW = [3,3,3,3]
    
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
        Xall, X, Xrepeat, Xexonic = fastReadGenome(fileName = fileName, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = outputAsDict_b,
#               outputGenomeString_b = outputGenomeString_b,
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
                    else:
                        Xenc[i] = codeW_asArray #wild-card array

                
                if outputGenomeString_b == 1:
                    return Xenc, XencRepeat, XencExonic,  X #, effectiveStartAtPosition, effectiveEndAtPosition

                else:
                    return Xenc, XencRepeat, XencExonic, 'genomeString_in_output_notCalled' #, effectiveStartAtPosition, effectiveEndAtPosition
                
                
            
            elif outputEncodedInt_b == 1:
    
                Xenc = np.zeros(shape = lenX, dtype = outputEncodedType)
                XencRepeat = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                XencExonic = np.zeros(shape = lenX, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                
                for i in range(lenX):
            
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
                    else:
                        Xenc[i] = codeW_asInt #int for wild-card 
                 
                if outputGenomeString_b == 1:
                    return Xenc, XencRepeat, XencExonic, X #, effectiveStartAtPosition, effectiveEndAtPosition
                    
                else:
                    return Xenc, XencRepeat, XencExonic, 'noGenomeStringCalled' #, effectiveStartAtPosition, effectiveEndAtPosition
        
        
        else:# outputEncoded_b == 0:
    
            return X, Xrepeat, Xexonic #, effectiveStartAtPosition, effectiveEndAtPosition
                                                
        
        
    #read in data for each chromosome and record it all in a dictionary:
    else: #outputAsDict_b != 0:
        
        
        #Read in data from genome
        XchrAllDict, XchrDict, XchrRepeatDict, XchrExonicDict, XchrNsDict = readGenome(fileName = fileName, 
               exonicInfoBinaryFileName = exonicInfoBinaryFileName,
               chromoNameBound = chromoNameBound, 
               startAtPosition = startAtPosition,
               endAtPosition = endAtPosition,
               outputAsDict_b = outputAsDict_b,
#               outputGenomeString_b = outputGenomeString_b,
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
                        else:
                            XencDict[chromo][i] = codeW_asArray #wild-card array

                            
            elif outputEncodedInt_b == 1:
                
                for chromo in XchrDict.keys():
                    
        #            lenXchrAll = XchrAll[chromo][0]
                    lenXchr = len(XchrDict[chromo])   
                    XencDict[chromo] = np.zeros(shape = lenXchr, dtype = outputEncodedType)
                    XencRepeatDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    XencExonicDict[chromo] = np.zeros(shape = lenXchr, dtype = 'int8') #'int' better than 'int8'? -- we only need a boolean
                    
                    for i in range(lenXchr):

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
                        else:
                            XencDict[chromo][i] = codeW_asInt #int for wild-card

                                
        
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


def getAllSamplesFromGenome(encodedGenomeData,
                            startPosition = 0,
                            endPosition = 1e26,
                            labelsCodetype = 0,
                            cutDownNrSamplesTo = 1e26,
                            genSamplesAtRandom_b = 0,
                            outputEncodedOneHot_b = 1,
                            outputEncodedInt_b = 0,
                            outputEncodedType = 'int8',
                            convertToPict_b = 0,
                            flankSize = 50, 
                            shuffle_b = 0, 
                            inner_b = 1, 
                            shuffleLength = 5,
                            augmentWithRevComplementary_b = 0):
    '''Get all "samples" of flanks and corr midpoints moving across a genome. Important: this will get the encoded bases at all positions, also
    those containing a not-ACGT letter! So to use the output, handling the non-qualified letters (encoded by the array for the wild card, W) 
    must be done in the application. 
    
    It is though possible to get samples at random positions (of the loaded genome data); for this set genSamplesAtRandom_b = 1 and use
    cutDownNrSamplesTo to set a number of samples lower than the length of the loaded genome data.
    
    Input:
    encodedGenomeData: encoded tuple dna-sequence, repeat info, exonic info as returned by encodeGenome and fastReadGenome
    
    OBS: startPosition and endPosition are relative to the input genomic sequence (in encodedGenomeData)! When running encodeGenome or fastReadGenome
    for generating this genomic sequence you also can use a start and end, but that's absolute positions in the genomic input seq to that run.
    
    Output: list X, Y, R, Q, sampledPositions where
    
    X: the set of left and right hand side flanks (of the set flankSize) for each position in the input (encoded genome seq)
    Y: the set of "labels" corr to X, ie the set of encoded bases for each position (: Y[i] is the base in the middle of the flanks X[i])
    R: boolean indicating whether the position is a repeat (1) or not (0)
    Q: boolean indicating whether the position is qulified (1) or not (0); a position, i, is qualified if the flanks X[i] and the base at i, Y[i],  consists
    in ACGT's only (so disqualified if containing a single "wild card" -- a not-ACGT)
    sampledPositions: an array containing all sampled positions (index in the encoded genome seq)

    '''
       
    if len(encodedGenomeData) == 4: #if encodeGenome was run with outputGenomeString_b =1:
        genomeSeq, repeatInfoSeq, exonicInfoSeq, genomeSeqAsString =  encodedGenomeData
    else:
        genomeSeq, repeatInfoSeq, exonicInfoSeq =  encodedGenomeData

    lenGenomeSeq = genomeSeq.shape[0]
    print("Length of encoded genome sequence read in: ", lenGenomeSeq)
    
    nrSamples = min(endPosition - max(flankSize, startPosition), min(lenGenomeSeq-flankSize, endPosition) - max(flankSize, startPosition))
    
    #cut down if wanted:
    nrSamples = int(min(cutDownNrSamplesTo, nrSamples))    

    if augmentWithRevComplementary_b == 1:
       #generate the reverse complemented sequence:
       complGenomeSeq = np.apply_along_axis(complementArray,1, genomeSeq)
       revComplGenomeSeq = complGenomeSeq[::-1]
       
#       print revComplGenomeSeq.shape
#       print genomeSeq[::-1][:10], revComplGenomeSeq[:10]
     
       nrSamples_aug = 2*nrSamples
       print "Nr of samples to be generated %d, but since we incl rev complements the total nr of samples is doubled %d" % (nrSamples, nrSamples_aug)
    
    else:        
        print("Nr of samples to be generated: ", nrSamples)
    
 
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
                Q = np.ones(shape = 2*nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
                
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = 2*nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
                
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize,4), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples, labelShape), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
            
    elif outputEncodedInt_b == 1:

        try:
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = 2*nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
    
        except MemoryError:
            
            nrSamples = 100000
            print("Due to memory limit I'll be reading in only the first %d samples" % nrSamples)
            
            if augmentWithRevComplementary_b == 1:
                X = np.zeros(shape = (2*nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (2*nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = 2*nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = 2*nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean?               
            else:
                X = np.zeros(shape = (nrSamples, 2*flankSize), dtype = outputEncodedType) #to hold the flanks
                Y = np.zeros(shape = (nrSamples), dtype = outputEncodedType) #to hold the labels
                R = np.zeros(shape = nrSamples, dtype = 'int8') #to hold the repeat info; int8 ok for boolean? 
                Q = np.ones(shape = nrSamples, dtype = 'int8') #to hold the qualified info; int8 ok for boolean? 
            
            
#    print "X shape", X.shape
#    print "Y shape", Y.shape
    
    #Loop along the genome and record the flanks and mid points:
    
    #If genSamplesAtRandom_b == 1, we first sample a set of positions and order them (this is to be able to compare with k-mer model in likelihood ratio tests only applied to a sub sample of the genomic seq)
    sampledPositions = np.zeros(nrSamples, dtype = 'int64') #to keep track of which positions were sampled; here the actual index in the genome seq
    sampledPositionsBoolean = np.zeros(lenGenomeSeq, dtype = 'int8') #to keep track of which positions were sampled; here a boolean indicator to make look-up easy afterwards
    if genSamplesAtRandom_b == 1:
        
        for k in range(nrSamples):
        
            idx = np.random.randint(max(flankSize, startPosition),min(lenGenomeSeq-flankSize, endPosition))
            
            sampledPositions[k] = idx
            sampledPositionsBoolean[idx] = 1
            
        sampledPositions.sort()
        print("sampledPositions ", sampledPositions)
        
    else: #all positions are "sampled"; we then only keep an indicator/boolean array of 1's, and leave the positions-array with zeros:

        sampledPositions = np.arange(max(flankSize, startPosition),min(lenGenomeSeq-flankSize, endPosition))        
        sampledPositionsBoolean = np.ones(lenGenomeSeq, dtype = 'int8')        
        #the first and last part both of length flankSize are not "sampled":
        for i in range(flankSize):
            sampledPositionsBoolean[i] = 0
            sampledPositionsBoolean[ lenGenomeSeq - 1 - i] = 0
        #Also, the positions up to startPosition and trailing endPosition are not included:
        for i in range(int(startPosition)):
            sampledPositionsBoolean[i] = 0
        for i in range(int(endPosition), lenGenomeSeq - 1):
            sampledPositionsBoolean[i] = 0    
            
        #        #the first part of length flankSize + startPosition are not "sampled":
#        for i in range(flankSize + startPosition):
#            sampledPositionsBoolean[i] = 0
#        #Also, the last positions from min(lenGenomeSeq-flankSize, endPosition) to lenGenomeSeq are not sampled
#        for i in range(int(min(lenGenomeSeq-flankSize, endPosition)), lenGenomeSeq):
#            sampledPositionsBoolean[i] = 0

    
    for i in range(nrSamples):
        
        if genSamplesAtRandom_b == 1:
            idx = sampledPositions[i] - flankSize #subtract flankSize as we fetch the mid-point/label from position idx + flankSize in the genome seq (see Y[i] below)
        else: #just use the position
            #the midpoint sampled is idx + flankSize (see below). Set:
            idx = i + int(max(flankSize, startPosition)) - flankSize 
            #Then:
            #1) when startPosition <= flankSize, idx + flankSize = i + flankSize = sampledPositions[i], 
            #   and so idx =  sampledPositions[i] - flankSize as in genSamplesAtRandom_b == 1 case
            #2) when startPosition > flankSize, idx + flankSize = i + startPosition = sampledPositions[i], 
            #   and again  idx = sampledPositions[i] - flankSize as in genSamplesAtRandom_b == 1 case

            
        #left flank:
#        for j in range(flankSize):
#            X[i][j] = genomeSeq[i + j] #i + j = i + firstSampleAt - flankSize + j
#        print("idx, genomeSeq[idx:(idx + flankSize)] " , idx, genomeSeq[idx:(idx + flankSize)])
        X[i][:flankSize] = genomeSeq[idx:(idx + flankSize)] #i + j = i + firstSampleAt - flankSize + j

        #right flank:
#        for j in range(flankSize + 1, 2*flankSize +1):
#            X[i][j-1] = genomeSeq[i + j]
        X[i][flankSize:] = genomeSeq[(idx + flankSize +1):(idx + 2*flankSize + 1)]
        
            
        if shuffle_b > 0.5:
#                print "I shuffle ..."
            if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                shuffleFlanks(v = X[i], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
            else:
                shuffle(X[i])          
   
        #labels at midpoint:
        R[i] = repeatInfoSeq[idx + flankSize]
        
        if labelsCodetype == 0:
            Y[i] = genomeSeq[idx + flankSize]
        elif labelsCodetype == 1:
            Y[i] = basePair(genomeSeq[idx + flankSize])
        elif labelsCodetype == -1:
            Y[i] = basePairType(genomeSeq[idx + flankSize])
        elif labelsCodetype == 2:
            if exonicInfoSeq[idx + flankSize] == 1:
                Y[i] = exonicInd
            elif repeatInfoSeq[idx + flankSize] == 1:
                Y[i] = repeatInd
            else:
                Y[i] = otherInd
        elif labelsCodetype == 3: #repeat or not?
            if repeatInfoSeq[idx + flankSize] == 1:
                Y[i] = repeatBinInd
            else:
                Y[i] = notRepeatBinInd
                
        #Determine whether the position is qualified or not, and record it in the assigned array:
        if np.max(X[i]) > 2 or np.max(Y[i]) > 2:
            Q[i] = 0
            
        #add reversed and complemented sequence if desired ---
        #by proceding exactly as for the non-transformed seq
        #only adding nrSamples to the sample index:
        if augmentWithRevComplementary_b == 1:
            
            #position on reverse strand corr to idx; this is then the end of the 
            #interval (context) that we want; the mid point of the desired context is then
            # at endPos - flankSize; the desired context then starts at endPos - 2flanksize:
            #to the left of that:
            endPos = lenGenomeSeq - 1 - idx
            idx = endPos - 2*flankSize
            
            #left flank:
    #        for j in range(flankSize):
    #            X[i][j] = genomeSeq[i + j] #i + j = i + firstSampleAt - flankSize + j
    #        print("idx, genomeSeq[idx:(idx + flankSize)] " , idx, genomeSeq[idx:(idx + flankSize)])
            X[i + nrSamples][:flankSize] = revComplGenomeSeq[idx:(idx + flankSize)] #i + j = i + firstSampleAt - flankSize + j
    
            #right flank:
    #        for j in range(flankSize + 1, 2*flankSize +1):
    #            X[i][j-1] = genomeSeq[i + j]
            X[i + nrSamples][flankSize:] = revComplGenomeSeq[(idx + flankSize +1):(idx + 2*flankSize + 1)]
            
                
            if shuffle_b > 0.5:
    #                print "I shuffle ..."
    #            print v2
                if inner_b == 0: #shuffle only the 2*shuffleLength long central part
                    shuffleFlanks(v = X[i + nrSamples], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                elif inner_b == 1:#shuffle only the outer left and right shuffleLength long outer parts
                    shuffleFlanks(v = X[i + nrSamples], flankSize = flankSize, shuffleLength = shuffleLength, inner_b = inner_b)
                else:
                    shuffle(X[i + nrSamples])          
       
            #labels at midpoint: 
            R[i+ nrSamples] = repeatInfoSeq[idx + flankSize]
            
            if labelsCodetype == 0:
                Y[i + nrSamples] = revComplGenomeSeq[idx + flankSize]
            elif labelsCodetype == 1:
                Y[i + nrSamples] = basePair(revComplGenomeSeq[idx + flankSize])
            elif labelsCodetype == -1:
                Y[i + nrSamples] = basePairType(revComplGenomeSeq[idx + flankSize])
            elif labelsCodetype == 2:
                if exonicInfoSeq[idx + flankSize] == 1:
                    Y[i + nrSamples] = exonicInd
                elif repeatInfoSeq[idx + flankSize] == 1:
                    Y[i + nrSamples] = repeatInd
                else:
                    Y[i + nrSamples] = otherInd
            elif labelsCodetype == 3: #repeat or not?
                if repeatInfoSeq[idx + flankSize] == 1:
                    Y[i + nrSamples] = repeatBinInd
                else:
                    Y[i + nrSamples] = notRepeatBinInd
                    
            #Determine whether the position is qualified or not, and record it in the assigned array:
            if np.max(X[i + nrSamples]) > 2 or np.max(Y[i  + nrSamples]) > 2:
                Q[i + nrSamples] = 0
                
            
#    #add reversed and complemented sequence: 
#    if augmentWithRevComplementary_b == 1:
        
#        xRev = X[:][::-1]
#        print "xRev.shape ", xRev.shape
##            xRevAug = np.asarray(list(map(complementArray, xRev)))
#        xRevAug = np.apply_along_axis(complementArray,2, xRev)
#        print "xRevAug.shape ", xRevAug.shape
#        yRevAug = np.apply_along_axis(complementArray,1, Y)
#        print "yRevAug.shape ", yRevAug.shape
#        for i in range(nrSamples):
#            X[i + nrSamples] = xRevAug[i]
#            Y[i + nrSamples ] = yRevAug[i]
        
    if outputEncodedOneHot_b == 1 and convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        
    return X, Y, R, Q, sampledPositions, sampledPositionsBoolean, augmentWithRevComplementary_b



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
    '''Generate a set of nrSamples samples. This can be done either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.
    
    labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1) 
                    or base pair type (purine/pyrimidine, -1); the prediction obtained will be of the
                    chosen code type (ie if 1 is used it is only the base pair at the given position which
                    is predicted). Pt only works with one-hot encoding.
    
    
    
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
    

    #NOT IN WORKING ORDER?
    if fromGenome_b == 0:
        
        i = 0
        for i in range(nrSamples):
            
            #generate random "letters", codeA, codeC, codeT, codeG
            for j in range(2*flankSize):
                idx = np.random.randint(0,4) 
                X[i][j][idx] = 1
                
            #generate random label:
            idx = np.random.randint(0,labelShape)
            Y[i][idx] = 1
    
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
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
                #Get random site unless we want to simply get the nrSamples samples from a running window of
                #length 2*flankSize across the selected part of the genome:
                if genRandomSamples_b == 0:
                    idx = flankSize + i
                    noQualifiedSampleObtainedYet_b = 1
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
         
                
                #In case we are sampling randomly and the sample contains a not-ACGT it must be discarded;
                #if we are "sampling" a stretch of the genome, we keep all samples --- so when using the output sample
                #the non-qulified letters/positions must be discarded there:  
                if genRandomSamples_b > 0:
                    
                    if np.max(X[i]) > 2 or np.max(Y[i]) > 2:
                        continue
                    else:
                        noQualifiedSampleObtainedYet_b = 1
                
                
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
                idx = np.random.randint(0,4)
                X[i][j][idx] = 1
                
            #generate random label:
            idx = np.random.randint(0,4)
            Y[i][idx] = 1
            
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
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
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
                
                #In case we are smapling randomly and the sample contains a not-ACGT it must be discarded;
                #if we are "sampling" a stretch of the genome, we keep all samples --- so when using the output sample
                #the non-qulified letters/positions must be discarded there:  
                if genRandomSamples_b > 0:
                    
                    if np.max(X[i]) > 2 or np.max(label) > 2:
                        continue
                    else:
                        noQualifiedSampleObtainedYet_b = 1
                
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
                                 indicatorArray,
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
                     augmentWithRevComplementary_b = 1,
                     testSampling_b = 0 #should only be used for small sample sizes 
                     ):
    '''Generate a set of nrSamples samples. This can be done either from an existing genome 
    (set fromGenome_b = 1 and supply a file name genomeFileName) or, with fromGenome_b = 0, by 
    sampling the flanks and midpoints at random (using np.random).
    
    Returns: X,Y where X is the sample of contexts and Y is the corresponding labels (ie midpoints of contexts; contexts do not inlcude the midpoints)  
             If testSampling_b: returns X, Y, Z where X, Y is as just described and Z is the list of indices of the sampled positions 
    
    indicatorArray: so as to confine the sampling to a set of indices; the indicatorArray should be 
    an array of booleans (0/1 int8) of lenght equal to genomeArray. Only relevant for fromGenome_b > 0, 
    genRandomSamples_b = 1 and getOnlyRepeats_b = 0 (ie the settings used when training a model -- the 
    getOnlyRepeats_b = 1 option is just there for handling a single experiment). 
    
    It is possible to shuffle the obtained flanks by setting shuffle_b = 1. With inner_b = 1
    only the central part of length 2*shuffleLength of the flanks are shuffled; with inner_b = 0
    only the left and right outer parts of length shuffleLength of the flanks are shuffled. If
    inner_b is not 0 or 1 (but shuffle_b =1) the flanks as a whole are shuffled.
    
    labelsCodetype: determines whether to encode the labels as bases (0 and default), base pairs (1), base 
                pair type (purine/pyrimidine, -1) or exonic/repeat/other (2); the prediction obtained will be of the
                chosen code type (ie if 1 is used it is only the base pair at the given position which
                is predicted). Pt only works with one-hot encoding.

    testSampling_b: if set to 1 the output will be append the list of sampled positions; fr this reason this 
                    option should only be used for small sample sizes (else long lists of long integers!)

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
        
        #for the purpose of testing that the sampling is uniform, include a list to 
        #hold the indexes of the sampled positions:
        if testSampling_b == 1:
            Z = np.zeros(shape = (nrSamples,1), dtype = 'int64' ) #MUST be int64; int32 covers only up to about 2.1 bill!

    #set a random seed bewteen 0 and 1e6:
#    np.random.seed(seed = np.random.randint(0,1e6))
    
    #NOT IN WORING ORDER!?
    if fromGenome_b == 0:
        
        i = 0
        for i in range(nrSamples):
            
            #generate random "letters", codeA, codeC, codeT, codeG
            for j in range(2*flankSize):
                idx = np.random.randint(0,4) 
                X[i][j][idx] = 1
                
                
            #generate random label:
            #NOT OK: should give a vector not an int
            idx = np.random.randint(0,labelShape) 
            Y[i][idx] = 1
    
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
        for i in range(nrSamples):
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
                #Get random site unless we want to simply get the nrSamples samples from a running window of
                #length 2*flankSize across the selected part of the genome:
                if genRandomSamples_b == 0:
                    idx = flankSize + i
                    noQualifiedSampleObtainedYet_b = 1
                else:
                    indexOk_b = 0
                    while  indexOk_b == 0:
                        idx = np.random.randint(flankSize+1,lGenome-flankSize)
                        if indicatorArray[idx] == 1:
                            indexOk_b = 1
    
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
                    
                    X[i][:flankSizeOut] = genomeArray[(idx - flankSize):(idx - exclFrqModelFlanks_b*flankSizeFrqModel)] #left-hand flank
                    X[i][(flankSizeOut+1):] = genomeArray[(idx+exclFrqModelFlanks_b*flankSizeFrqModel+1):(idx + 1 + flankSize)] #right-hand flank
                    
                    
                    
                #In case we are sampling randomly and the sample contains a not-ACGT it must be discarded;
                #if we are "sampling" a stretch of the genome, we keep all samples --- so when using the output sample
                #the non-qualified letters/positions must be discarded there:  
                if genRandomSamples_b > 0:
                    
                    if np.max(X[i]) > 2 or np.max(Y[i]) > 2:
                        continue
                    else:
                        noQualifiedSampleObtainedYet_b = 1
                        
                        
                #if we want to include the output from a frq-model (k-mer model) in our model, we place the input of the frq-model at the center of X (ie at flankSizeOut):
                if inclFrqModel_b == 1 and outputEncodedOneHot_b == 1:
                    
                    #before 21 Aug 2019:
    #                try:
    #                    X[i][flankSizeOut] = frqModelDict[genString]
    #                except KeyError:
    #                    prin( "KeyError when reading from frqModelDict, key: ", genString)
    #                    X[i][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                    #After 21 Aug 2019:
                    #we fetched genString above
                    try:
                        X[i][flankSizeOut] = np.log(frqModelDict[genString]) - np.log([0.25, 0.25, 0.25, 0.25]) #reason for centering: get both neg and pos numbers
                    except KeyError:
                        print( "KeyError when reading from frqModelDict, key: ", genString)
                        X[i][flankSizeOut] = np.array([0.0, 0.0, 0.0, 0.0]) #np.log([0.25, 0.25, 0.25, 0.25]) #np.array([0.0, 0.0, 0.0, 0.0])
                
                
                    
                
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
                            #before 21 Aug 2019:
    #                        try:
    #                            X[i + nrSamples][flankSizeOut] = frqModelDict[revComplGenString]
    #                        except KeyError:
    #                            X[i + nrSamples][flankSizeOut] = [0.25, 0.25, 0.25, 0.25]
                            #After 21 Aug 2019:
                            try:
                                X[i + nrSamples][flankSizeOut] = np.log(frqModelDict[revComplGenString])
                            except KeyError:
                                X[i + nrSamples][flankSizeOut] = np.log([0.25, 0.25, 0.25, 0.25])
                            
                    
                            
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
            
            #for the purpose of testing that the sampling is uniform:
            if testSampling_b == 1:
                
                Z[i][0] = idx
            
    if outputEncodedOneHot_b == 1 and convertToPict_b == 1:
        
        X = seqToPict(inputArray = X, flankSize = flankSize)
        

     
#    if transformStyle_b == 0:  
    if testSampling_b == 1:
        
        return Z, Z
        
    else: #the typical use case
    
        return X, Y  
    



#labelsCodetype NOT ENCORPORATED
#Generally no in working order
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
        for i in range(nrSamples):
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
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
                
                #In case we are smapling randomly and the sample contains a not-ACGT it must be discarded;
                #if we are "sampling" a stretch of the genome, we keep all samples --- so when using the output sample
                #the non-qulified letters/positions must be discarded there:  
                if genRandomSamples_b > 0:
                    
                    if np.max(X[i]) > 2 or np.max(label) > 2:
                        continue
                    else:
                        noQualifiedSampleObtainedYet_b = 1
                
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
        for i in range(nrSamples):
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
                #Get random site 
                idx = np.random.randint(flankSize+1,lGenome-flankSize) 
        
        
                #If only repeat-positions are wanted:
                if getOnlyRepeats_b == 1:
                    
                    while repeatArray[idx] == 0:
                        
                        idx = np.random.randint(flankSize+1,lGenome-flankSize)
                        
                #In case the sample contains a not-ACGT it must be discarded;
                genString = genomeString[(idx - flankSize):(idx + 1 + flankSize)]
                if genString.count('W') > 0:
                    continue
                else:
                    noQualifiedSampleObtainedYet_b = 1
                        
                    
                    
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
        for i in range(nrSamples):
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
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
                
                   
                #In case  the sample contains a not-ACGT it must be discarded (and we must fetch a new one);  
                if np.max(X[i]) > 2 or np.max(Y[i]) > 2:
                    continue
                else:
                    noQualifiedSampleObtainedYet_b = 1
                    
                
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
        for i in range(nrSamples):
            
            noQualifiedSampleObtainedYet_b = 0
            while noQualifiedSampleObtainedYet_b == 0:
            
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
                
                
                #In case  the sample contains a not-ACGT it must be discarded (and we must fetch a new one);  
                if np.max(X[i]) > 2 or np.max(Y[i]) > 2:
                    continue
                else:
                    noQualifiedSampleObtainedYet_b = 1
    
                
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
    


######################################################################################
## Read genes info
######################################################################################


def readRefGene(refGeneFile,rootOutput, chromoPrefix = 'hg38_', omitGenesPrefixed = ['MIR', 'RNU'], chromoOrderList =[]):
    '''Read RefGene file from UCSC. Format of input:
    http://genome.ucsc.edu/cgi-bin/hgTables
    
    Output: 
    -- gene-dict mapping gene name to chromo, txStart, txEnd, score, strand, exonCnt
    -- isoformDict mapping [chromo][geneName][transStart] to a list of 'isoforms', each a tuple [txStart, txEnd, score, strand, exonCnt]     
    
    Dumps these files in rootOutput folder.
    '''
        
    if not(chromoOrderList):
        chromoOrderList_useThis = []
    else:
        chromoOrderList_useThis = chromoOrderList
        
    #We first get hold of all "genes along with their number of possible isoforms"; a "gene" is here taken  
    #to mean a pair (gene id, transcription start) and all records sharing such a pair represent the various
    #isoforms of that gene. As we subsequently want to pick out only one gene representative per gene id, we keep the 
    #number of the isoforms. In the step to follow we then take for each gene id the "isoform" (transription start)
    #having the largest isoform count 
    isoformDict = {}
    fileReader = open(refGeneFile, "r")
#    fileReader = csv.reader(refGeneFile, delimiter='\t')
    lineCnt = 0
    for row in fileReader:

        row = row.strip()        
        row = row.split('\t')
        
        if lineCnt < 3:
            print(row)
        
        if lineCnt < 1: 
            lineCnt += 1
            continue
            
        chromo =  chromoPrefix + row[2]
        strand = row[3]
        txStart = row[4].strip() #transcription start
        txEnd = row[5].strip() #... end (Obs: txEnd is transcription start if strand is -)
        cdsStart = row[6].strip() #coding region start
        cdsEnd = row[7].strip() #... end
        exonCnt = row[8].strip()
        score = row[11].strip()
        geneName = row[12]
    
            
        #Only include chomos in the list:
        if chromoOrderList and not(chromo in chromoOrderList):
            lineCnt += 1
            continue
        
        #Omit genes having any of the named prefixes to be omitted:
        if geneName[:3] in omitGenesPrefixed: 
            lineCnt += 1
            continue
        
        if strand == '+':
            transStart =txStart
        elif strand == '-':
            transStart = txEnd    
                 
        
        if not(chromo in isoformDict):
                
            isoformDict[chromo] = {}
            
            if not(chromo in chromoOrderList_useThis):
                chromoOrderList_useThis.append(chromo)
        
        if geneName in isoformDict[chromo]:
            
            if transStart in isoformDict[chromo][geneName]:
                
                isoformDict[chromo][geneName][transStart].append([txStart, txEnd, score, strand, exonCnt] )#one more occ of this isoform (ie this gene with this transcription start site) 
            
            else:
                
                isoformDict[chromo][geneName][transStart] = [[txStart, txEnd, score, strand, exonCnt]] #first entry rep's the isoform count
        
        else:
            
            isoformDict[chromo][geneName] = {}
            isoformDict[chromo][geneName][transStart] = [[txStart, txEnd, score, strand, exonCnt]] #first entry rep's the isoform count
 
        lineCnt += 1
        
        
    #Now: take for each gene id the version (transcr Start) that has the largest number of isoforms:    
    outputDict = {}
    for chromo in isoformDict:
       
        if not(chromo in outputDict):
                
            outputDict[chromo] = {}
            
            if not(chromo in chromoOrderList_useThis):
                chromoOrderList_useThis.append(chromo)
            
        for geneName in isoformDict[chromo]:
            
            maxIsoCount = -1
#            maxIsoCountAtStart = -1e15
            
            for transStart in isoformDict[chromo][geneName]:
                
                listOfIsos = isoformDict[chromo][geneName][transStart]
                
                isoCount = len(listOfIsos)
                
                maxTrLength = -1
                
#                isoCount = isoformDict[chromo][geneName][transStart][0]
                if isoCount >=  maxIsoCount:
                    
                    maxIsoCount = isoCount  
                    
                    #loop through iso's and find the one having the longest "transcr length":                    
                    for iso in listOfIsos: 
                        
                        txStart, txEnd, score, strand, exonCnt = iso
                        trLength = np.abs(int(txEnd) - int(txStart)) #abs since strand dependency ...
                        if trLength > maxTrLength:
                            maxTrLength = trLength
                            outputDict[chromo][geneName] = iso
                            
            txStart, txEnd, score, strand, exonCnt = outputDict[chromo][geneName]
                            
               
    #Next: dump the isoformDict and outputDict content to file:
    dumpFile = rootOutput + 'refGeneDict.p'
    pickle.dump(outputDict, open(dumpFile, "wb"))
    dumpFile = rootOutput + 'isoFormDict.p'
    pickle.dump(isoformDict, open(dumpFile, "wb"))  
    
    return outputDict, isoformDict




    
######################################################################################
### Generators for sampling in the NN-taining and testing
######################################################################################    
#
#                    
#def myGenerator_MLP(customFlankSize,
#                    batchSize, 
#                    inclFrqModel_b, 
#                    labelsCodetype, 
#                    forTrain_b,                 
#                    genomeArray, 
#                    repeatArray,
#                    exonicArray,
#                    trainSetIndicator,
#                    testSetIndicator,
#                    getOnlyRepeats_b,
#                    genomeString,
#                    frqModelDict,
#                    flankSizeFrqModel,
#                    exclFrqModelFlanks_b,
#                    outputEncodedOneHot_b,
#                    outputEncodedInt_b,
#                    shuffle_b,
#                    inner_b,
#                    shuffleLength,
#                    augmentWithRevComplementary_b, 
#                    Xconv =  np.asarray([]),
#                    XfrqConv = np.asarray([])
#                    ):
#                          
#    while 1:
#        
#        if forTrain_b == 1: #for train set
#            X,Y = genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = trainSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
#                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
#        else: #for test set        
#            X,Y = genSamplesForDynamicSampling_I(nrSamples = batchSize, genomeArray = genomeArray, repeatArray = repeatArray, exonicArray = exonicArray, indicatorArray = testSetIndicator, flankSize = customFlankSize, getOnlyRepeats_b = getOnlyRepeats_b, inclFrqModel_b = inclFrqModel_b,
#                             genomeString = genomeString, frqModelDict = frqModelDict, flankSizeFrqModel = flankSizeFrqModel, exclFrqModelFlanks_b = exclFrqModelFlanks_b, outputEncodedOneHot_b = outputEncodedOneHot_b, labelsCodetype = labelsCodetype, outputEncodedInt_b = outputEncodedInt_b, shuffle_b = shuffle_b , inner_b = inner_b, shuffleLength = shuffleLength, augmentWithRevComplementary_b = augmentWithRevComplementary_b)
#
#        if inclFrqModel_b == 1:
#
#            Xconv[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :] = X[:, :(customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel), :]
#            Xconv[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel):, :] = X[:, (customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel +1):, :]
#            XfrqConv[:, 0, :] = X[:,customFlankSize - exclFrqModelFlanks_b*flankSizeFrqModel, :]
#            
#            Xmlp  = Xconv.reshape((Xconv.shape[0],Xconv.shape[1]*Xconv.shape[2]))
#            Xfrq =  Xconv.reshape((XfrqConv.shape[0],XfrqConv.shape[1]*XfrqConv.shape[2]))
#            
#            yield([Xfrq, Xmlp],Y)
#    
#        
#        elif inclFrqModel_b == 0:
#            
#            Xmlp = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
#
#            
#            yield(Xmlp,Y)

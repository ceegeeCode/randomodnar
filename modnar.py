#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:03:04 2019

@author: Christian Grønbæk
"""

#import numpy

import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import sys
sys.path.append('/isdata/kroghgrp/tkj375/various_python/DNA_proj/development/')

from dnaNet_dataGen import readGenome, genSamplesForDynamicSampling_I, intLetter, invIntLetter, invNumpyIntInt, genSamplesDirectlyFromGenome

@registry.register_problem
class randomodnar(text_problems.Text2ClassProblem):
    
    """... dna randomness ... """
    
    @property
    def approx_vocab_size(self):
        return 4   

    @property
    def num_classes(self):
        return 4
     
#    @property
#    def class_labels(self, data_dir):
##      del data_dir
#      return ["ID_%d" % i for i in range(self.num_classes)]
##       return ['A', 'C', 'T','G']
   
    @property
    def is_generate_per_split(self):
      # generate_data will shard the data into TRAIN and EVAL for us.
      return False 

    @property
    def dataset_splits(self):
      """Splits of data to produce and number of output shards for each."""
      # 10% evaluation data
      return [{
          "split": problem.DatasetSplit.TRAIN,
          "shards": 9,
      }, {
          "split": problem.DatasetSplit.EVAL,
          "shards": 1,
      }]
    

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split
        
        rootGenome = r"/isdata/kroghgrp/tkj375/data/DNA/human/hg19/"
#        fileName = r"hg19.fa"
        fileName = r"hg19.fa"
        fileGenome = rootGenome +fileName
            
    
        exonicInfoBinaryFileName = ''
        
        startAtPosition = 1000000
        endAtPosition  = 200000000
        flankSize = 100
        
        #We use a int-encoding here, since we are going to trnaform it back to letters anyhow (below)
        outputAsString_b = 1
        outputEncodedOneHot_b = 0
        outputEncodedInt_b = 0
        outDtype = 'int8'
        
        genomeArray, genomeString , repeatArray, exonicArray= readGenome(fileName = fileGenome, exonicInfoBinaryFileName  = exonicInfoBinaryFileName , startAtPosition = startAtPosition, endAtPosition = endAtPosition, outputGenomeString_b = 1, outputAsDict_b = 0)
        print("genome string first 100:", genomeArray[:100])
#        input("Sille Sille ... du er skoer Topper")
        lGenome = len(genomeString)
        genomeSeqSourceTrain = 'Read data from whole genome (chromo\'s concatenated, if any)'
        
        nrSamples= 10000000
        
        X, Y = genSamplesDirectlyFromGenome(genomeString = genomeString,
                                 nrSamples = nrSamples,
                                 augmentWithRevComplementary_b = 0,
                                 flankSize = flankSize, 
                                 outputAsString_b = outputAsString_b,
                                 outputEncodedOneHot_b = outputEncodedOneHot_b,
                                 outputEncodedInt_b = outputEncodedInt_b,
                                 labelsCodetype = 0,
               outputEncodedType = outDtype,
               getOnlyRepeats_b = 0,
               repeatArray = '',
               shuffle_b = 0,
               shuffleLength = 0,
               inner_b = 0)
        
#        X, Y = genSamplesForDynamicSampling_I(nrSamples = nrSamples,
#                                     genomeArray = genomeArray, 
#                                     repeatArray = repeatArray ,
#                                     exonicArray = exonicArray ,
#                                     transformStyle_b = 0,
#                                     X = 0,
#                                     Y = 0,
#                                     labelsCodetype = 0,
#                                     fromGenome_b = 1, 
#                                     inclFrqModel_b = 0,
#                                     frqModelDict = {},
#                                     flankSizeFrqModel = 4,
#                                     exclFrqModelFlanks_b = 0,
#                                     genomeString = '', 
#                         outDtype = outDtype,
#                         getOnlyRepeats_b = 0,
#                         genRandomSamples_b = 1,
#                         outputEncodedOneHot_b = outputEncodedOneHot_b,
#                         outputEncodedInt_b = outputEncodedInt_b,
#                         convertToPict_b = 0,
#                         flankSize = 50, 
#                         shuffle_b = 0, 
#                         inner_b = 1, 
#                         shuffleLength = 5, 
#                         getFrq_b = 0,
#                         augmentWithRevComplementary_b = 0
#                         )
        
        
        stepSize = 1
        for i in range(nrSamples):
            if i%100000 == 0:
                print("Done ", i, " samples")
#            print(i,Y[i])
#            inpList = map(invIntLetter, X[i])
            #we make a list of overlapping or of disjoint words:
            #overlapping:
#            inp = ''
#            for j in range(len(X[i])-5):
#                for k in range(5):
#                    inp += X[i][(j+k):(j+1+k)]
##                    inp += invIntLetter(X[i][(j+k):(j+1+k)])
#                inp += ' '
            
            #disjoint:
            inp = ''
            for j in range(0, len(X[i]),stepSize):
                inp += X[i][j:(j+stepSize)]
#                    inp += invIntLetter(X[i][(j+k):(j+1+k)])
                inp += ' '
                
#            out = invIntLetter(Y[i])
            out = intLetter(Y[i]) #class labels should be integers, not letters
#            print(inp, out)
#            out = invNumpyIntInt(Y[i])
#            print(out)
#            out = Y[i]
            
            yield {
                  "inputs": inp,
                  "label": out
              }


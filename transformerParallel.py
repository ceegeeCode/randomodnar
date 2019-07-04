#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:18:16 2019

@author: Christian Grønbæk

"""

from __future__ import absolute_import, division, print_function

from tensor2tensor.utils import registry
from tensor2tensor import models

import tensorflow as tf

from tensorflow import estimator

from tensor2tensor.bin import t2t_datagen


import modnar

import multiprocessing as mp

import os

import json

from tensor2tensor.utils.trainer_lib import create_hparams

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment

from tensorflow.python.tools import inspect_checkpoint as chkp


import subprocess as sub
#import subprocess as sub2

HOME_DNA_PROJ=r'/isdata/kroghgrp/tkj375/various_python/DNA_proj'


PROBLEM = 'randomodnar'

MODEL = 'transformer_encoder'
#HPARAMS = 'transformer_base_single_gpu'
HPARAMS = 'transformer_tiny'
HPARAMS_BATCHSIZE = 'batch_size=4096'

USR_DIR=HOME_DNA_PROJ + '/development'


TMP_DIR_1 = HOME_DNA_PROJ +  '/scratch/t2t_datagen_1'
#TMP_DIR_1 = '{$HOMEDAT/scratch/t2t_datagen}' # Where data files from internet stored
DATA_DIR_1 = HOME_DNA_PROJ + '/t2t_data_1' # Where pre-prcessed data is stored
#DATA_DIR_1 = '{$HOME_DNA_PROJ/t2t_data_1}' # Where pre-prcessed data is stored

TMP_DIR_2 = HOME_DNA_PROJ +  '/scratch/t2t_datagen_2'
#TMP_DIR_1 = '{$HOMEDAT/scratch/t2t_datagen}' # Where data files from internet stored
DATA_DIR_2 = HOME_DNA_PROJ + '/t2t_data_2' # Where pre-prcessed data is stored
#DATA_DIR_1 = '{$HOME_DNA_PROJ/t2t_data_1}' # Where pre-prcessed data is stored

TRAIN_DIR = HOME_DNA_PROJ + '/t2t_train/' + PROBLEM + r'/' + MODEL + '-' + HPARAMS


vocabName = 'vocab.randomodnar.1024.subwords'
VOCAB = HOME_DNA_PROJ + '/t2t_vocabs/' + vocabName

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR


# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Chngaes to Hparams
hparams.batch_size = 1024
#hparams.learning_rate_warmup_steps = 45000
#hparams.learning_rate = .4

# Can see all Hparams with code below
#print(json.loads(hparams.to_json()))


trainSteps = 2500
trainStepsIncr = 2500
evalSteps = 10

replaceVocab_b = 0
#problemInst = modnar.randomodnar()



def runThis(gridPoint):
    
    n, m = gridPoint
    
    print("At n, m ", n, m)
    
    if n%2 == 0: 
        
#        problemInst = modnar.randomodnar()
        
        if m == 0: #generate data and place them in data_dir2:
        
            #Remove data dir 2:
            sub.call(['rm', '-r', DATA_DIR_2] )
            
            #Create a directory if its not already there:
            if not os.path.exists(DATA_DIR_2):
    
                os.makedirs(DATA_DIR_2)
                print("Directory " + DATA_DIR_2 + " created ")
                
            else:
                
                print("Directory " + DATA_DIR_2 + " already exists or could not be created")


            #Create a directory if its not already there:
            if not os.path.exists(TMP_DIR_2):
    
                os.makedirs(TMP_DIR_2)
                print ("Directory " + TMP_DIR_2 + " created ")
                
            else:
                
                print ("Directory " + TMP_DIR_2 + " already exists or could not be created")
                
                
#            #Clear the directory:
#            sub.call(['rm', DATA_DIR_2 + '/' + r'*'] )
#            print("Cleared the data dir ",DATA_DIR_2 )
#            sub.call(['ls', DATA_DIR_2] )
             
            #Generate new data
            args = ['t2t-datagen', '--t2t_usr_dir=' + USR_DIR, '--problem=' + PROBLEM,'--data_dir=' +  DATA_DIR_2, '--tmp_dir=' + TMP_DIR_2 ]
#            + " --usr_dir=" + USR_DIR + " --data_dir=" +  DATA_DIR_2 + " --tmp_dir=" + TMP_DIR_2
            print(args)
            sub.call(args)
            
            if replaceVocab_b == 1:
                #replace the generated vocab by the standard one:
                sub.call(['rm', DATA_DIR_2 + r'/' + vocabName]) 
                sub.call(['cp', VOCAB, DATA_DIR_2 + r'/' + vocabName])
            
            print("Replaced the generated vocab with the std one")


#            t2t_datagen.main(problem = PROBLEM, data_dir = DATA_DIR_2, tmp_dir = TMP_DIR_2)
            
        
        if m == 1: #train model on data in data_dir1:
            
            trainSteps = 2500 + n*2500 #trainSteps + trainStepsIncr
            evalSteps = 100 #evalSteps
            
            
#            # Initi Run COnfig for Model Training
#            RUN_CONFIG = estimator.RunConfig(
#                  model_dir=TRAIN_DIR # Location of where model file is store
##                  model_name=PROBLEM
#                  # More Params here in this fucntion for controling how often to save checkpoints and more. 
#            )
#            
#            # Create Tensorflow Experiment Object
#            tensorflow_exp_fn = create_experiment(
#                    run_config=RUN_CONFIG,
#                    hparams=hparams,
#                    model_name=MODEL,
#                    problem_name=PROBLEM,
#                    data_dir=DATA_DIR_1, 
#                    train_steps=trainSteps, # Total number of train steps for all Epochs
#                    eval_steps=evalSteps # Number of steps to perform for each evaluation
#                )
#            
##            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#            
#            tensorflow_exp_fn.train_and_evaluate()
            
            argsTrain = ['t2t-trainer', '--t2t_usr_dir=' + USR_DIR, '--problem=' + PROBLEM,'--data_dir=' +  DATA_DIR_1, 
                         '--model=' + MODEL, '--hparams_set=' + HPARAMS, 
                         '--hparams=' + HPARAMS_BATCHSIZE,
                         '--output_dir='+TRAIN_DIR, 
                         '--schedule=train_and_evaluate',
                         '--train_steps=' + str(trainSteps), 
                         '--eval_steps=' + str(evalSteps)]
#                         '--schedule=continuous_eval']
            print(argsTrain)
            sub.call(argsTrain)
                
                    
    else:
        
#        problemInst = modnar.randomodnar()
        
        #Remove data dir 1:
        sub.call(['rm', '-r', DATA_DIR_1] )
        
        if m == 0: #generate data and place them in data_dir1:
            
            #Create a directory if its not already there:
            if not os.path.exists(DATA_DIR_1):
    
                os.makedirs(DATA_DIR_1)
                print("Directory " + DATA_DIR_1 + " created ")
                
            else:
                
                print("Directory " + DATA_DIR_1 + " already exists or could not be created")


            #Create a directory if its not already there:
            if not os.path.exists(TMP_DIR_1):
    
                os.makedirs(TMP_DIR_1)
                print("Directory " + TMP_DIR_1 + " created ")
                
            else:
                
                print("Directory " + TMP_DIR_1 + " already exists or could not be created")

#            #Clear the directory:
#            sub.call(['rm', DATA_DIR_1 + '/' + r'*'] )
#            print("Cleared the data dir ",DATA_DIR_1 )
#            sub.call(['ls', DATA_DIR_1] )
            
            #Generate new data
            argsData = ['t2t-datagen', '--t2t_usr_dir=' + USR_DIR, '--problem=' + PROBLEM,'--data_dir=' +  DATA_DIR_1, '--tmp_dir=' + TMP_DIR_1 ]
            print(argsData)
            sub.call(argsData)
            
            #replace the generated vocab by the standard one:
            if replaceVocab_b == 1:
                sub.call(['rm', DATA_DIR_1 + r'/' + vocabName]) 
                sub.call(['cp', VOCAB, DATA_DIR_1 + r'/' + vocabName])

            print("Replaced the generated vocab with the std one")

        
        if m == 1: #train model on data in data_dir_2:
            
            trainSteps = 2500 + n*2500 #trainSteps + trainStepsIncr
            evalSteps = 100 #evalSteps
            
            
#            # Initi Run COnfig for Model Training
#            RUN_CONFIG = estimator.RunConfig(
#                  model_dir=TRAIN_DIR # Location of where model file is store
##                  model_name=PROBLEM
#                  # More Params here in this fucntion for controling how often to save checkpoints and more. 
#            )
#            
#            # Create Tensorflow Experiment Object
#            tensorflow_exp_fn = create_experiment(
#                    run_config=RUN_CONFIG,
#                    hparams=hparams,
#                    model_name=MODEL,
#                    problem_name=PROBLEM,
#                    data_dir=DATA_DIR_2, 
#                    train_steps=trainSteps, # Total number of train steps for all Epochs
#                    eval_steps=evalSteps # Number of steps to perform for each evaluation
#                )
#            
##            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#            
#            tensorflow_exp_fn.train_and_evaluate()
            
            argsTrain = ['t2t-trainer', '--t2t_usr_dir=' + USR_DIR, '--problem=' + PROBLEM,'--data_dir=' +  DATA_DIR_2, 
                         '--model=' + MODEL, '--hparams_set=' + HPARAMS, 
                         '--hparams=' + HPARAMS_BATCHSIZE,
                         '--output_dir='+TRAIN_DIR, 
                         '--schedule=train_and_evaluate',
                         '--train_steps=' + str(trainSteps), 
                         '--eval_steps=' + str(evalSteps)]
#                         '--schedule=' + 'continuous_eval']
            print(argsTrain)
            sub.call(argsTrain)
                


def runThisN0(m):
    runThis(n=0,m=m)


def runThisN1(m):
    runThis(n=1,m=m)


#Step0: generate dat in data_dir_1 for the first training to run at:
runThis((1,0))

# Step 1: Init multiprocessing.Pool()
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(2)

# Step 2: # Kick off Training
for n in range(100):
    
    
    results = pool.map(runThis, [(n,m) for m in range(2)])
    
#    if n%2 == 0:
#        results = pool.map(runThisN0, [m for m in range(2)])
#        
##    chkp.print_tensors_in_checkpoint_file("/isdata/kroghgrp/tkj375/various_python/DNA_proj/t2t_train/randomodnar/transformer_encoder-transformer_base_single_gpu/model.", tensor_name='', all_tensors=True)
#
#    if n%2 == 1:
#        results = pool.map(runThisN1, [m for m in range(2)])
        
#    chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

    
#    runThis(n,0)
#    results = [pool.apply(runThis, args=(n,m)) for m in range(2)]

# Step 3: Don't forget to close
pool.close()    
            
        


                
                
                
                
                
    

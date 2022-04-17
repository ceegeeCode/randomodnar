
This repository contains code  -- all in a 'research state' -- that was used for the paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 

To train the neural networks models with this code you'll need Keras and Tensorflow (and more set-up if you want to run it 
on a GPU, which you probably do ... it takes seting up some connection between your GPU and Tensorflow). 

The code for the paper was written in Python 2.7. It is placed in the py27 folder. For the neural networks part we used Tensorflow (1.12) with Keras (2.1.6) as backend. We used also matplotlib, numpy, scipy. You have maybe also noticed the py36 folder: It contains some of the modules of the py27 folder, but modified so as to run in Python 3.6. The py36 folder also contains a Jupyter Notebook with a walk-through of an example (see below).     

### Setup your environment, for instance:

Assuming `conda`:

`conda create -n dna python=3.6`
`source activate dna`
`conda install matplotlib`
`pip install tensorflow-gpu keras`


### py27 and py36 modules

In each of the Python module there is some description of what that module does -- and generally a lot of 
concrete calls to the functions defined in the code. It is written for our own research work, so not user
friendly. It will therefore probably take time to follow what is going on. But it is possible to simply copy a 
selection of lines in on of the 'Usage' sections and run it by pasting it into a python console. There is though 
a Notebook in the py36 folder (see below), which is intended to be readable :--). 

Very short descriptions of the main modules:

dnaNet_LSTM, dnaNet_CONV, dnaNet_MLP: modules for seting up and training networks

dnaNet_dataGen: module for various data handling (reading in DNA strings, 1-hot encoding, sampling)

dnaNet_stats: big modul in which a model can be applied ('prediction') and performance can be investigated (plots, tables, LR tests, Fourier transformation)

### py36 Notebook

In the py36 folder there is a Jupyter Notebook, LSTM50yeast.ipynb, containing an example of applying our code to handle the case of training and applying a LSTM50 model on the yeast genome. We ran it in Google Colab using a Google drive to store the code and the results. But you can of course run it in your own installation.   

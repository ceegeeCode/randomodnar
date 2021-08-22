
This repository contains code  -- all in a 'research state' -- that was used for the paper:

C.Grønbæk, Y.Liang, D.Elliott, A.Krogh, "Prediction of DNA from context using neural
networks", July 2021, bioRxiv, doi: https://doi.org/10.1101/2021.07.28.454211.

Please cite the paper if you use the code -- or parts of it -- in your own work. 

To train the neural networks models in this code you'll need Keras and Tensorflow (and more set-up if you want to run it 
on a GPU, which you probably do ... it takes some connection between your GPU and Tensorflow). 

The code is written for Python 2.7. We use also matplotlib, numpy, scipy.

In each of the Python module there is some description on what that module does -- and generally a lot of 
concrete calls to the functions defined in the code. It is written for our own research work, so not user
friendly. So it will probalby take time to follow what is going on. But it is possible to simply copy a 
selection of lines in on of the 'Usage' sections and run it by pasting it into a python console (e.g. in a
Jupyter Notebook).

Veryn short descriptions of the main modules:
dnaNet_LSTM, dnaNet_CONV, dnaNet_MLP: modules for seting up and training networks
dnaNet_dataGen: module for various data handling (reading in DNA strings, 1-hot encoding, sampling)
dnaNet_stats: big modul in which a model can be applyed ('prediction') and performance can be investigated (plots, tables, LR tests, Fourier transformation)


# TheanoNN
Material layers recognition from Elipsometrical spectra by Theano based NN. This is a demo of Mutly Layer Perceptron Neural Networl based on Theano library in python2.7.

As an input the Real Part of an amplitude of reflected light depend on angle is ised (250 points).
An output is a thickness and Re and Im of refraction coefficient.

The launch is:

python control.py

By defauls each time of lauching the NN coefficients are renewing. But you can switch it off and proceed to train the same one each time of running if you don't change the NN architecture.

control.py - The main precess

MLP.py - the class of NN

Layer.py - the calss of layer

Lagerraum.py - ("storage" - germ.) Just a file with two functions: loading the data and adapting to the format understandable for NN. And the second function (a bit of mess): creation of actual cost functions for training and validatino. Here also block of visualisation os located. The visualisation works on-line!!!

Some demo pictures (If you launch the program - they will be rewritten!):

CostEvol.png - evolution of the cost function during a work. "Cost" - for training cost function, "Valid" - for validaiton.

LearnCoefEvol.png - the training method is gradient decent. Here adaptive learning coefficient is ised. It is modified proportionaly to the gradient of cost function. This is the evelution depend on epoch.

OutputParamsMatching.jpg - the comparison of output parameters of a well trained NN for training samples.


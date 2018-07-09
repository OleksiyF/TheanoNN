from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
import os
import sys
import matplotlib.pyplot as plt

rng = numpy.random

class Layer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        print "Init the Layer"
        
        self.input = input

        self.w = theano.shared(rng.randn(n_in, n_out).astype(theano.config.floatX), name="w")
        self.b = theano.shared(rng.randn(n_out).astype(theano.config.floatX), name="b")
	#print "initW\n",self.w.get_value()
	#print "initb\n",self.b.get_value()
        
        self.params_t = [self.b]
        self.params = [self.w, self.b]
        self.output = self.Process()
        
    def cost_fcn(self, y):
	#print self.output
        return T.mean((self.output-y)**4)

    def Process(self):
         return 1 / (1 + T.exp(-T.dot(self.input, self.w)-self.b))


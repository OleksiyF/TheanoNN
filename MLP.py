from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
import os
import sys
import matplotlib.pyplot as plt
from Layer import Layer
import os.path

rng = numpy.random

class MLP(object):
    def __init__(self, map):#n_in, n_hidden, n_out):
       
       print "Init the MLP"
       
       self.x = T.matrix('x')
       self.y = T.matrix('y')
       #input = self.x
       input_to_next_layer=self.x
       
       self.layer_num=len(map)-1
       
       self.hiddenLayer=[]
       #input_to_next_layer=input
       for i in range(self.layer_num):
           self.hiddenLayer.append(Layer(
                                         input=input_to_next_layer,
                                         n_in=map[i],
                                         n_out=map[i+1],
                                         activation=T.tanh
                                         )   
                                   )
           input_to_next_layer=self.hiddenLayer[-1].output
           
       self.cost_fcn = self.hiddenLayer[-1].cost_fcn
       
    def Process(self):
        for layer in self.hiddenLayer:
            layer.Process()
        return self.hiddenLayer[-1].output
    
    def Save(self, id):
        print "Saving the NN"
        for ilayer in range(len(self.hiddenLayer)):
            numpy.savetxt( './data/'+id+str(ilayer)+'w.txt',  self.hiddenLayer[ilayer].w.get_value())
            numpy.savetxt( './data/'+id+str(ilayer)+'b.txt',  self.hiddenLayer[ilayer].b.get_value())
        print "NN has been saved"    
        #numpy.savetxt( './data/'+id+'last'+'w.txt',  self.hiddenLayer[-1].w.get_value())
        #numpy.savetxt( './data/'+id+'last'+'b.txt',  self.hiddenLayer[-1].b.get_value())

    def Load(self, id, renew=False):
	if not os.path.isfile('./data/'+id+str(0)+'w.txt'):
	    print "File does not exist. Init new weights."
	    return False
	elif renew==True:
	    print "Manualy init new weights."
	    return False
		
        
	print "Loading the NN..."        
        for ilayer in range(len(self.hiddenLayer)): 
            data_w = numpy.loadtxt('./data/'+id+str(ilayer)+'w.txt')
            data_b = numpy.loadtxt('./data/'+id+str(ilayer)+'b.txt')
	    
		#In case of one newron there is a some fuck
#	    print "HERE", data_w
	    if data_w.shape!=self.hiddenLayer[ilayer].w.get_value().shape:
		data_w=numpy.transpose([data_w])
                data_b = numpy.array([data_b])

            data_w = data_w.astype(theano.config.floatX)
	    self.hiddenLayer[ilayer].w.set_value(data_w)
            data_b = data_b.astype(theano.config.floatX)
	    self.hiddenLayer[ilayer].b.set_value(data_b)
        
    def PrintNN(self):
        for layer in self.hiddenLayer:
            print "w"
            print layer.w.get_value()
            print "b"
            print layer.b.get_value()

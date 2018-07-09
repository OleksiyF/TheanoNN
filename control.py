from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
import os
import sys
import matplotlib.pyplot as plt
from Lagerraum import *

#import GPU testing function
from GPUtest import GPUtestd

###############
###uncoment the next block with "gpu0" option if you want to enable gpu parallelisation
###But be aware that you need CUDA libraries and a good graphics card :)
###############
#init GPU
#import theano.sandbox.cuda
from numpy import dtype, float32
#theano.sandbox.cuda.use(device="gpu0")
#theano.sandbox.cuda.use(device="cpu")

#setup float32 usage
config.floatX = 'float32'
#setup saving garbedge
#config.allow_gc = False
#config.optimizer="None"
config.exception_verbosity="high"
print "Device =", config.device
print "Garbedge collection =", config.allow_gc
print "FloatX =", config.floatX
#GPUtestd()
rng = numpy.random

#Select hidden layer size
#n_neu=600	#0.0032
#n_neu=800	#0.0032
#n_neu=200	#0.0032
n_neu=300	#0.0032

#the number of over all samples (training+validation)
#train_size=300
#train_size=1200
#train_size=130*20
train_size=130


n_epochs=1000000        # nubmer of epoch
learn_limit=0.0000001   # cost vlue bottom limit
learn_rate=0.0001       # coeff that controls the speed of learning coefficient adaption depend of gradient of cost fcn

is_train = True         #to train or not to train? Legacy
#is_train = False

#is_trade = True        #Legacy from Stock excahnge version of the program
is_trade = False

is_renew_weights = True     #Do i want to renew NN coefficients is load them if exist
#is_renew_weights = False
#is_loadNN= True
#is_loadNN= False

###Load the data
data, target, valid_data, valid_target, insize, outsize= load_dataset(size=train_size)

##Choose the architecture of NN
#map=[insize, n_neu, outsize]
#map=[insize, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, n_neu, outsize]
map=[insize, n_neu, n_neu, n_neu, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, n_neu, n_neu, n_neu, outsize]

#print target
print "Map:", map

#Create name of the NN for saving and loading
NNid=''
for i in map:
    NNid = NNid + str(i) + '.'

### create NN
NN = MLP(map=map)
#if is_loadNN==True:

#Load the data
NN.Load(NNid, is_renew_weights)

#init theano
f = theano.function([NN.x], [NN.Process()])
res_plot = plt.subplot(212)
if is_train == True:
    print "Train"
    print "Train length", len(target)
    print "Validation length", len(valid_target)
    ##traninig
    #plt.ion()
    costval, validval = test_mlp(data, target, valid_data, valid_target, \
                       NNid, NN,learning_rate=learn_rate, learning_limit=learn_limit, n_epochs=n_epochs)
    NN.Save(NNid)
    
    NN.x.tag.test_value = data
    #f = theano.function([NN.x], [NN.Process()])
    prediction = f(data)[0]  

#    plt.ioff()
#    plt.show()	

#Training is finished

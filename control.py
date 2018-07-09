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

#n_neu=600	#0.0032
#n_neu=800	#0.0032
n_neu=200	#0.0032

#train_size=300
#train_size=1200
#train_size=130
#train_size=130*20
train_size=130

#n_epochs=1000000        
n_epochs=1000000        
learn_limit=0.0000001
learn_rate=0.0001

is_train = True
#is_train = False

#is_trade = True
is_trade = False

is_renew_weights = True
is_renew_weights = False
#is_loadNN= True
#is_loadNN= False

data, target, valid_data, valid_target, insize, outsize= load_dataset(size=train_size)
#map=[insize, n_neu, outsize]
#map=[insize, n_neu, n_neu, outsize]
map=[insize, n_neu, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, n_neu, n_neu, outsize]
#map=[insize, n_neu, n_neu, n_neu, n_neu, n_neu, n_neu, outsize]

#print target
print "Map:", map

NNid=''
for i in map:
    NNid = NNid + str(i) + '.'

NN = MLP(map=map)
#if is_loadNN==True:
NN.Load(NNid, is_renew_weights)
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

#Training is finished. Lets predict some thing

def Predict(init_data):
    
    #create proper dimension
    #print len(init_data), memory
    init_data = init_data.reshape(1,memory*mem_factor)
    
    #for i in xrange(prediction_per_step):
    if True:
        #predict value
        NN.x.tag.test_value = init_data
        predicted_value=f(init_data)[0][0]
        #print predicted_value
        #load the value to array with predictions
        #fight_predicted.append(predicted_value[0])
        #replace input tadata with new value
        #go = numpy.delete(go, 0, 1)
        #go = numpy.append(go, predicted_value.reshape(1,1), axis=1)
        
    return predicted_value
        
#fight_predicted = Predict(prediction_per_step, fight_data[0])
#position = 80400
#fight_predicted = Predict(prediction_per_step, data[position])

#threshold1 = 0.0006
#threshold2 = 0.0006
threshold=0.00035
price = 0.
saldo = 0.
pos_deal = False
neg_deal = False
#for prediction_per_step in numpy.linspace(10, 50, 3):

plt.ion()
pred_fig = plt.figure(4)
pred_mon = pred_fig.add_subplot(111)
pred_mon.set_autoscaley_on(True) 

'''
trade_data = fight_data
fight_predicted = Predict(trade_data[0])
pred_mon.clear()
x_pred = numpy.linspace(len(trade_data[0]), len(trade_data[0])+len(fight_predicted), len(fight_predicted))
pred_mon.legend()
for i in range(0,prediction_per_step):
    x_data = numpy.linspace(0, len(trade_data[i]), len(trade_data[i]))
    pred_mon.plot(x_data, trade_data[+1], 'g',label='data')
pred_mon.plot(x_pred, fight_predicted, 'r',label='predict')
#pred_mon.plot(x, x*lin_par[0]+lin_par[1], 'b',label='fit')
pred_mon.relim()
pred_mon.autoscale_view()
pred_fig.canvas.draw()
plt.show()
'''
#for threshold in numpy.linspace(0.001,0.007,1):
if True:
#    slope_data=[]
#    saldo=0
    if is_trade==True:
	print "Trade simulation:"
        #for i in range(0,len(data)/2):
        trade_data = fight_data
        for i in range(0,len(trade_data)):
            if i%100==0 and i !=0:
                print i
	    #print len(trade_data[i])
            fight_predicted = Predict(trade_data[i])
            x_pred = numpy.linspace(len(trade_data[i])+i, len(trade_data[i])+i+len(fight_predicted), len(fight_predicted))
            x_data = numpy.linspace(i, len(trade_data[i])+i, len(trade_data[i]))
            #lin_par = numpy.polyfit(x, fight_predicted, 1)
            #slope_data.append(lin_par[0])
        
	    if i<len(trade_data)-1:
                #pred_mon.clear()
                pred_mon.plot(x_pred, fight_predicted, 'r',label='predict')
                pred_mon.plot(x_data, trade_data[i], 'g',label='data')
#                pred_mon.plot(x, x*lin_par[0]+lin_par[1], 'b',label='fit')
 #               pred_mon.legend()
                pred_mon.relim()
                pred_mon.autoscale_view()
                pred_fig.canvas.draw()
	

	#    raw_input("Enter some thing")
            '''
	    #Buy
            if lin_par[0] > threshold and pos_deal==False and neg_deal==False:
                price = trade_data[i][-1]
                pos_deal = True
                #print "Buy.pos_deal.", price
            if lin_par[0] < -threshold and pos_deal==False and neg_deal==False:
                price = trade_data[i][-1]
                neg_deal = True
                #print "Buy.neg_deal.", price
            #Sell
            if lin_par[0] < threshold and pos_deal==True:
                saldo = saldo + trade_data[i][-1]-price
                pos_deal = False
                #print "Sell.pos_deal.", data[i][-1], "Delta", data[i][-1]-price
            if lin_par[0] > -threshold and neg_deal==True:
                saldo = saldo + price-trade_data[i][-1]
                neg_deal = False
                #print "Sell.neg_deal.", data[i][-1], "Delta", price-data[i][-1]
                
        print "RESULT saldo", saldo, "for threshold", threshold'''

#hist,bins = numpy.histogram(slope_data,50)
#center = (bins[:-1] + bins[1:]) / 2
#width = 0.7 * (bins[1] - bins[0])
#plt.bar(center, hist, align='center', width=width)

#plt.plot(numpy.linspace(0,len(slope_data),len(slope_data)),slope_data)
    
#x = numpy.linspace(len(data), len(fight_predicted)+len(data), len(fight_predicted))
#lin_par = numpy.polyfit(x, fight_predicted, 1)
#res_plot.plot(x ,lin_par[0]*x+lin_par[1], 'b', label='Fit')

#res_plot.plot(x, fight_predicted, 'k', label='Fight_pred')

#compare = data[position+memory]
#res_plot.plot(numpy.linspace(len(data), len(compare)+len(data), len(compare)), 
#              compare, 'g', label='original data')

#plt.plot([1,2],[1,3])
print "Done fight"  
#res_plot.legend()
#plt.show(block=True)
plt.show()

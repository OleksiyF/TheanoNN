from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
import os
import sys
import matplotlib.pyplot as plt
from MLP import MLP
from get_stock_data import *
import sys

rng = numpy.random

def load_dataset( name_spec='mindata.txt', name_indata="mytextfile.txt", size=0):
    
    file_spec = open(name_spec)
 
    #size=200000
    size_limitator=0

    maxlen=0
    data_spec=[]
    counter=0
    for string in file_spec:
        counter=counter+1
        if counter%7==1:
            #splited=string.split()
            data_spec=string.split()
	    #if maxlen<len(splited):	
	#	maxlen=len(splited)
#		print "Maximum lenght is", maxlen
	    #minimum=[]	
	    #for cell in splited:
	#	minimum.append(float(cell))	
            #data_spec.append(minimum)
	   
	    #size of data limitation	
	    size_limitator=size_limitator+1
	    if size_limitator>=size and size!=0:
		break
'''
	#conmlement all samplet to one length. add zero to missing part
    for i in range(len(data_spec)):
	add_zeros=maxlen-len(data_spec[i])
	#print add_zeros
	for j in range(add_zeros):
	    data_spec[i]= data_spec[i] + [0] 
'''
# load indata
    file_indata = open(name_indata)
  
    size_limitator=0
    data_indata=[]
    indata_cell=[]
    counter=0
    for string in file_indata:
        counter=counter+1

        if counter==13: 
	    data_indata.append(indata_cell)

	    #size of data limitation	
	    size_limitator=size_limitator+1
	    if size_limitator>=size and size!=0:
	    	break 
	    
	    counter=0
	    indata_cell=[]

        if counter==2 or counter==3 or counter==11 or counter==12 or counter==13 or counter==0:
	    continue
	indata_cell.append(float(string))	
    print "Size of indata", len(data_indata), "; Size of numimums", len(data_spec)
    if len(data_indata) != len(data_spec):
	sys.exit("len(data_indata) != len(data_spec)") 	

#normalisation of indata 
    data_indata = numpy.array(data_indata)
    for column in range(len(data_indata[0])):
    	print "max",max(data_indata[:,column])
	if max(data_indata[:,column])==0:
	    continue 
	data_indata[:,column]=data_indata[:,column]/max(data_indata[:,column])
    	#rint data_indata[:,column]
    data_indata = data_indata.astype(theano.config.floatX)

#normalisation of outdata 
    data_spec = numpy.array(data_spec)
    for column in range(len(data_spec[0])):
    	print "max",max(data_spec[:,column])
	if max(data_spec[:,column])==0:
	    continue 
	data_spec[:,column]=data_spec[:,column]/max(data_spec[:,column])
    	#rint data_indata[:,column]
    data_spec = data_spec.astype(theano.config.floatX)

    return data_spec[0:len(data_indata)*3/4], data_indata[0:len(data_indata)*3/4], data_spec[len(data_indata)*3/4:len(data_indata)], data_indata[len(data_indata)*3/4:len(data_indata)], maxlen 

def test_mlp(data, target, valid_data, valid_target, \
             NNid, classifier, learning_rate=0.01, learning_limit=0.01, n_epochs=1000, L1_reg=0.00, L2_reg=0.0001):
 
    D=[data, target]
    B=[valid_data, valid_target]
    
    cost = classifier.cost_fcn(classifier.y) #+ L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    precessed_data = classifier.hiddenLayer[-1].output
    
    params=[]
    for layer in classifier.hiddenLayer:
        params=params+layer.params

    aftergrad_params = T.grad(cost, params)

    upd = []   
    for param_i, grad_i in zip(params, aftergrad_params):
        upd.append((param_i, param_i - learning_rate*grad_i))
    
    train = theano.function(
            inputs=[classifier.x,classifier.y],
            outputs=[cost, precessed_data],
            updates=upd,
            name = "train")
    
    valid = theano.function(
            inputs=[classifier.x,classifier.y],
            outputs=[cost, precessed_data],
            name = "valid")
    
    
    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            train.maker.fgraph.toposort()]):
        print 'Used the cpu'
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              train.maker.fgraph.toposort()]):
        print 'Used the gpu'
    else:
        print 'ERROR, not able to tell if theano used the cpu or the gpu'
        print train.maker.fgraph.toposort()
    
    costval=[]
    validval=[]
    stop_lim=True
    counter = 0
    lowest_valid=99999
    save_freq=200
    
    plt.ion()

    fig1 = plt.figure(4, figsize=(20,10))
    res_mon1 = fig1.add_subplot(111)
    res_mon1.set_autoscaley_on(True) 
    
    fig2 = plt.figure(2)
    tr_mon = fig2.add_subplot(111)
    tr_mon.set_autoscaley_on(True) 
    
    fig3 = plt.figure(3)
    res_mon = fig3.add_subplot(111)
    res_mon.set_autoscaley_on(True) 
   
    intime = time.time()

    #print D[0], D[1]	

    while counter<n_epochs-1 and stop_lim:
        train_val, train_proc = train(D[0], D[1])
        valid_val, valid_proc = valid(B[0], B[1])

#	print "trainval ",train_val
	
        if lowest_valid>valid_val:
	    lowest_valid=valid_val

        costval.append(train_val)
        validval.append(valid_val)

        if counter%save_freq==0 and counter !=0 :
            if validval[-1]<=lowest_valid or True:
                classifier.Save(NNid)
 		#print "bebe"
            else:
                print "Validation is not improuving. NN has not been saved."
            
            tr_mon.clear()
        #    tr_mon.plot(numpy.linspace(0, len(validval), len(validval)), validval,'r',label='valid')
            tr_mon.plot(numpy.linspace(0, len(costval), len(costval)), costval, 'b',label='cost')
            tr_mon.legend()
            tr_mon.relim()
            tr_mon.autoscale_view()
            fig2.canvas.draw()

	    res_mon1.clear()
            res_mon1.plot(numpy.linspace(0, len(D[0]), len(D[1])), D[1], 'r',label='train')
            #res_mon1.plot(numpy.linspace(0, len(D[0]), len(D[1])), D[0], 'k',label='train')
            res_mon1.plot(numpy.linspace(0, len(train_proc), len(train_proc)), train_proc, 'b',label='train_proc')
            #res_mon.legend()
            res_mon1.relim()
            res_mon1.autoscale_view()
            fig1.canvas.draw()
#	    
            res_mon.clear()
            res_mon.plot(numpy.linspace(0, len(B[1]), len(B[1])), B[1], 'r',label='valid')
            res_mon.plot(numpy.linspace(0, len(valid_proc), len(valid_proc)), valid_proc, 'b',label='valid_proc')
            #res_mon.legend()
            res_mon.relim()
            res_mon.autoscale_view()
            fig3.canvas.draw()
#	    plt.ioff()
#	    plt.show()	
            print counter, "epoch; Costval is", train_val, "Validval is", valid_val, "Limit is", learning_limit
        if counter>2:
            stop_lim = costval[counter]>learning_limit# and validval[-2]>=validval[-1]
        #if stop_lim==False or (counter%999==0 and counter !=0):
            #print counter+1, "epoch; Costval is", train_val[0], "Limit is", learning_limit
        counter+=1
    print "Training time is", time.time()-intime, "Finish on the ", counter, "epoch with ", train_val, " cost"
    #plt.show(block=True)
    return costval, validval

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
import os
import sys
import matplotlib.pyplot as plt
from MLP import MLP
# from get_stock_data import *
import sys

rng = numpy.random


# Loading the in and out data
def load_dataset(name_spec='rpdata.txt', name_indata="mytextfile.txt", size=0):
    file_spec = open(name_spec)

    # size=200000
    size_limitator = 0

    maxlen = 0
    data_spec = []
    counter = 1
    for string in file_spec:
        counter = counter + 1
        if counter % 7 == 0:
            splited = string.split()
            one_spectrum = []
            for cell in splited:
                one_spectrum.append(float(cell))
            data_spec.append(one_spectrum)

            # size of data limitation
            size_limitator = size_limitator + 1
            if size_limitator >= size and size != 0:
                break

    # Complement all samples to one length. add zero to missing part
    '''
    for i in range(len(data_spec)):
    add_zeros=maxlen-len(data_spec[i])
    for j in range(add_zeros):
        data_spec[i]= data_spec[i] + [0]
    '''
    # Load indata
    file_indata = open(name_indata)

    size_limitator = 0
    data_indata = []
    indata_cell = []
    counter = 0
    for string in file_indata:
        counter = counter + 1

        if counter == 13:
            data_indata.append(indata_cell)

            # Size of data limitation
            size_limitator = size_limitator + 1
            if size_limitator >= size and size != 0:
                break

            counter = 0
            indata_cell = []

        if counter == 2 or counter == 3 or counter == 11 \
                or counter == 12 or counter == 13 or counter == 0:
            continue
        indata_cell.append(float(string))
    print "Size of indata", len(data_indata), "; Size of numimums", len(data_spec)
    if len(data_indata) != len(data_spec):
        sys.exit("len(data_indata) != len(data_spec)")

    data_indata = numpy.array(data_indata)
    data_spec = numpy.array(data_spec)

    # Delete equal rows
    ''''
    prev=numpy.array([-999])
    index_to_del=[]
    for row in range(len(data_spec)):
        if numpy.all(prev==[-999]):
            prev=data_spec[row]
            continue
        if numpy.all([prev==data_spec[row]]):
            #prev==data_indata
            #data_indata = numpy.delete(data_indata, row, axis=0)
            #data_spec = numpy.delete(data_spec, row, axis=0)
            index_to_del.append(row)
        else:
            prev=data_spec[row]

        for row in range(len(index_to_del)):
            data_indata = numpy.delete(data_indata, index_to_del,0)
            data_spec = numpy.delete(data_spec, index_to_del,0)

    print "Result data length", len(data_spec)
    '''
    maximum = 0
    # Normalisation of indata
    for column in range(len(data_indata[0])):
        print "max", column, max(data_indata[:, column])
        if max(data_indata[:, column]) == 0:
            continue
        if column == 0 or column == 1 or column == 4 or column == 7:
            maximum = 2.5
        else:
            maximum = max(data_indata[:, column])

        data_indata[:, column] = data_indata[:, column] / maximum
    # rint data_indata[:,column]
    data_indata = data_indata.astype(theano.config.floatX)

    # Normalisation of outdata
    maximum = 0
    for column in range(len(data_spec[0])):
        #    	print "max", column, max(data_spec[:,column])

        if max(data_spec[:, column]) == 0:
            continue
        if max(data_spec[:, column]) > maximum:
            maximum = max(data_spec[:, column])

    for column in range(len(data_spec[0])):
        data_spec[:, column] = data_spec[:, column] / maximum
    # rint data_indata[:,column]
    data_spec = data_spec.astype(theano.config.floatX)

    # data, target, valid_data, valid_target, insize, outsize
    return data_spec[0:len(data_indata) * 3 / 4], \
           data_indata[0:len(data_indata) * 3 / 4], \
           data_spec[len(data_indata) * 3 / 4:len(data_indata)], \
           data_indata[len(data_indata) * 3 / 4:len(data_indata)], \
           len(data_spec[0]), \
           len(data_indata[0])  # outsize


def test_mlp(data, target, valid_data, valid_target,
             NNid, classifier, learning_rate=0.01, learning_limit=0.01,
             n_epochs=1000, L1_reg=0.00, L2_reg=0.0001):
    D = [data, target]
    B = [valid_data, valid_target]

    cost = classifier.cost_fcn(classifier.y)  # + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    precessed_data = classifier.hiddenLayer[-1].output

    params = []
    for layer in classifier.hiddenLayer:
        params = params + layer.params

    aftergrad_params = T.grad(cost, params)

    upd = []

    learning_coef_sym = T.scalar('learning_coef_sym', dtype=config.floatX)
    #    learning_coef_sym.tag.test_value = 0.01

    # Let's do this!
    for param_i, grad_i in zip(params, aftergrad_params):
        upd.append((param_i, param_i - learning_coef_sym * grad_i))

    # Creation of the train function
    train = theano.function(
        inputs=[classifier.x, classifier.y, learning_coef_sym],
        outputs=[cost, precessed_data],
        updates=upd,
        name="train")

    # Creation of the validation function
    valid = theano.function(
        inputs=[classifier.x, classifier.y],
        outputs=[cost, precessed_data],
        name="valid")

    # Copy paste from somewhere :) Tell if you use gpu or not
    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            train.maker.fgraph.toposort()]):
        print 'Used the cpu'
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              train.maker.fgraph.toposort()]):
        print 'Used the gpu'
    else:
        print 'ERROR, not able to tell if theano used the cpu or the gpu'
        train.maker.fgraph.toposort()

    costval = []
    validval = []
    learning_coef_hist_data = []
    stop_lim = True
    counter = 0
    lowest_valid = 99999
    save_freq = 100

    # visual=False
    visual = True
    plt.ion()

    # Do you want to visualise the data?
    if visual == True:
        fig1 = plt.figure(4, figsize=(20, 10))
        res_mon1 = fig1.add_subplot(111)
        res_mon1.set_autoscaley_on(True)

        fig2 = plt.figure(2)
        tr_mon = fig2.add_subplot(111)
        tr_mon.set_autoscaley_on(True)

        fig3 = plt.figure(3)
        res_mon = fig3.add_subplot(111)
        res_mon.set_autoscaley_on(True)

        fig4 = plt.figure(5)
        coef_mon = fig4.add_subplot(111)
        coef_mon.set_autoscaley_on(True)

    intime = time.time()

    learning_coef = 0.01
    while counter < n_epochs - 1 and stop_lim:
        train_val, train_proc = train(D[0], D[1], learning_coef)
        valid_val, valid_proc = valid(B[0], B[1])

        # Recalculate the learning coeff
        if counter > 25:
            learning_coef = float(
                train_val / (numpy.fabs(numpy.mean(costval[-20:-1]) - train_val)) * learning_rate
            )
            if learning_coef < 0.01:
                learning_coef = 0.01
            if learning_coef > 30:
                learning_coef = 30
            # print "learning coef", learning_coef
            learning_coef_hist_data.append(learning_coef)

        if lowest_valid > valid_val:
            lowest_valid = valid_val

        costval.append(train_val)
        validval.append(valid_val)

        if counter % save_freq == 0 and counter != 0:
            if validval[-1] <= lowest_valid or True:
                classifier.Save(NNid)  # SAVE THE NN
            else:
                print "Validation is not improuving. NN has not been saved."
            if visual == True:
                # Cost function for training and validation.
                tr_mon.clear()
                # tr_mon.plot(numpy.linspace(0, len(validval), len(validval)), validval,'r',label='valid')
                tr_mon.semilogy(numpy.linspace(0, len(validval), len(validval)), validval, 'r', label='valid')
                tr_mon.semilogy(numpy.linspace(0, len(costval), len(costval)), costval, 'b', label='cost')
                tr_mon.grid(True)
                tr_mon.legend()
                tr_mon.relim()
                tr_mon.autoscale_view()
                tr_mon.set_xlabel('Iterations')
                tr_mon.set_ylabel('Cost value')
                fig2.canvas.draw()
                fig2.savefig("CostEvol")

                # Comparison of output parameters for training data
                res_mon1.clear()
                res_mon1.plot(numpy.linspace(0, len(D[0]), len(D[1])), D[1], 'r', label='train')
                # res_mon1.plot(numpy.linspace(0, len(D[0]), len(D[1])), D[0], 'k',label='train')
                res_mon1.plot(numpy.linspace(0, len(train_proc), len(train_proc)), train_proc, 'b', label='train_proc')
                # res_mon.legend()
                res_mon1.relim()
                res_mon1.autoscale_view()
                fig1.canvas.draw()

                # Comparison of output parameters for validation data
                res_mon.clear()
                res_mon.plot(numpy.linspace(0, len(B[1]), len(B[1])), B[1], 'r', label='valid')
                res_mon.plot(numpy.linspace(0, len(valid_proc), len(valid_proc)), valid_proc, 'b', label='valid_proc')
                # res_mon.legend()
                res_mon.relim()
                res_mon.autoscale_view()
                fig3.canvas.draw()

                # Evolution of training coefficient
                coef_mon.clear()

                coef_mon.semilogy(numpy.linspace(0, len(learning_coef_hist_data), len(learning_coef_hist_data)),
                                  learning_coef_hist_data, 'b', label='learning coef')
                # coef_mon.hist(learning_coef_hist_data, 100)
                coef_mon.autoscale_view()
                coef_mon.grid(True)
                coef_mon.set_xlabel('Iterations')
                coef_mon.set_ylabel('Learning Coef value')
                fig4.canvas.draw()
                fig4.savefig("LearnCoefEvol")

            print  counter, "epoch; Costval is", train_val, "Validval is", valid_val, "Limit is", learning_limit
        if counter > 2:
            stop_lim = costval[counter] > learning_limit

        counter += 1
    print "Training time is", time.time() - intime, "Finish on the ", \
    counter, "epoch with ", train_val, " cost"
    # plt.show(block=True)
    return costval, validval

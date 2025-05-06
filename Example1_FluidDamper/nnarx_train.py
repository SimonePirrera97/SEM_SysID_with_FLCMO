"""MIMO_NNARX_identification

Simone Pirrera

Jan 22, 2022

## Training of a NNARX network to perform nonlinear system identification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from statistics import mean

## Import data
import scipy.io as sio

# Load the .mat file
mat_contents = sio.loadmat("data//data_es2.mat")
utrain = mat_contents['u_train']
ytrain = mat_contents['y_train']
Ntrain = len(utrain)
utrain = utrain[0:Ntrain,:]
ytrain = ytrain[0:Ntrain,:]

dim_u = utrain.shape[1]
dim_y = ytrain.shape[1]

#mat_contents = sio.loadmat(PATH_DATA + "//dataset_test_03.mat")
u_test = mat_contents['u_valid']
y_test = mat_contents['y_valid']

order = 4
in_shape = order*dim_y+(order+1)*dim_u
x_train = np.zeros((Ntrain-order,in_shape))
y_train = np.zeros((Ntrain-order,dim_y))
for k in range(order,Ntrain):
    phi = np.concatenate( [np.concatenate([utrain[k-t,:] for t in range(order+1)]), np.concatenate([ytrain[k-t-1,:] for t in range(order)])] )
    x_train[k-order,:] = phi    

y_train = ytrain[order:,:]

num_layers = 1 #2 #1 

outfile = open("results//results_nnarx_order3_6neur.txt", 'a')

for run in range(10):
    for num_neurons in [6]: #[3,5,7,10]:  #[5,8,10,15]:
    ## Build RNN

        print(f"### Number of neurons: {num_neurons}", file=outfile)
        print(f"### Number of neurons: {num_neurons}")
        
        model = keras.Sequential()
        model.add(layers.Dense(activation='tanh', units=num_neurons, input_shape = [in_shape] ))
        if num_layers > 1:
            model.add(layers.Dense(activation='tanh', units=num_neurons ))
        if num_layers > 2:
            model.add(layers.Dense(activation='tanh', units=num_neurons ))
        model.add(layers.Dense(activation='linear', units=dim_y))
        model.build()
        model.summary()
        
        #tot_param = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        #print(f"Trainable parameters: {tot_param}", file=outfile)
        #print(f"Trainable parameters: {tot_param}")

        # Model compilation
        model.compile(loss = 'mse', optimizer = 'adam')
        # Model training 
        starttime = time.time()
        model.fit(x_train, y_train, epochs = 2000, verbose=2)
        endtime = time.time()
        elapsed_time = endtime - starttime
        print(f"Elapsed time: {elapsed_time} seconds", file=outfile)
        #print(f"Elapsed time: {elapsed_time} seconds")
     
        ## Test model and print results
        utest = u_test  #np.reshape(u_test,(1,u_test.shape[0],dim_u))
        ytest = y_test
        Ntest = u_test.shape[0]
        
        # Extract model matrices: single layer
        W1 = model.get_weights()[0]
        b1 = model.get_weights()[1]
        if num_layers == 1:
            C = model.get_weights()[2]
            d = model.get_weights()[3]
        elif num_layers == 2:
            # Extract model matrices: two layers
            W2 = model.get_weights()[2]
            b2 = model.get_weights()[3]
            C = model.get_weights()[4]
            d = model.get_weights()[5]
        elif num_layers == 3:
            # Extract model matrices: three layers
            W2 = model.get_weights()[2]
            b2 = model.get_weights()[3]
            W3 = model.get_weights()[4]
            b3 = model.get_weights()[5]
            C = model.get_weights()[6]
            d = model.get_weights()[7]
        
        # Perform simulation 
        ysim_train = np.zeros((Ntrain,dim_y))
        ysim_train[0:order,:] = ytrain[0:order,:]
        ysim_test = np.zeros((Ntest,dim_y))
        ysim_test[0:order,:] = ytest[0:order,:]
        for k in range(order,Ntrain):
            phi = np.concatenate( [np.concatenate([utrain[k-t,:] for t in range(order+1)]), np.concatenate([ysim_train[k-t-1,:] for t in range(order)])] )
            if num_layers == 1:
                yp = np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), C ) + d   # single layer
            elif num_layers == 2:
                yp = np.matmul( np.tanh( np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), W2) + b2), C ) + d  # two layers
            elif num_layers == 3:
                yp = np.matmul( np.tanh( np.matmul( np.tanh( np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), W2) + b2), W3) + b3), C ) + d  # three layers
            ysim_train[k,:] = yp
        for k in range(order,Ntest):
            phi = np.concatenate( [np.concatenate([utest[k-t,:] for t in range(order+1)]), np.concatenate([ysim_test[k-t-1,:] for t in range(order)])] )
            if num_layers == 1:
                yp = np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), C ) + d   # single layer
            elif num_layers == 2:
                yp = np.matmul( np.tanh( np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), W2) + b2), C ) + d  # two layers
            elif num_layers == 3:
                yp = np.matmul( np.tanh( np.matmul( np.tanh( np.matmul( np.tanh( np.matmul(phi,W1) + b1 ), W2) + b2), W3) + b3), C ) + d  # three layers
            ysim_test[k,:] = yp
        
        # t = [k for k in range(Ntest)];
        # plt.plot(t,y_test)
        # plt.plot(t,ysim_test)
        # plt.show()
        
        ypred_train = model.predict(x_train)
        
        x_test_pred = np.zeros((Ntest-order,in_shape))
        y_test_pred = np.zeros((Ntest-order,dim_y))
        for k in range(order,Ntest):
            phi = np.concatenate( [np.concatenate([utest[k-t,:] for t in range(order+1)]), np.concatenate([ytest[k-t-1,:] for t in range(order)])] )
            x_test_pred[k-order,:] = phi    
        y_test_pred = ytest[order:,:]
        ypred_test = model.predict(x_test_pred)

        # rmse1 = math.sqrt(sum((ytrain[:,0] - ysim_train[:,0])**2))/math.sqrt(Ntrain)
        # rmse2 = math.sqrt(sum((ytrain[:,1] - ysim_train[:,1])**2))/math.sqrt(Ntrain)
        # print('training rmse = ', rmse1, rmse2, file=outfile)
        # print('training rmse = ', rmse1, rmse2)
        # rmse1 = math.sqrt(sum((y_train[:,0] - ypred_train[:,0])**2))/math.sqrt(Ntrain)
        # rmse2 = math.sqrt(sum((y_train[:,1] - ypred_train[:,1])**2))/math.sqrt(Ntrain)
        # print('training rmse (prediction) = ', rmse1, rmse2, file=outfile)
        # print('training rmse (prediction) = ', rmse1, rmse2)

        mse1 = (sum((y_test[:,0] - ysim_test[:,0])**2))/(2*Ntest)
        #rmse2 = math.sqrt(sum((y_test[:,1] - ysim_test[:,1])**2))/math.sqrt(Ntest)
        #print('test rmse = ', rmse1, rmse2, file=outfile)
        print('test rmse = ', mse1, file=outfile)
        #print('test rmse = ', rmse1, rmse2)
        fit1 = 100*(1- math.sqrt(sum(np.power(y_test[:,0]-ysim_test[:,0],2))) / math.sqrt(sum(np.power(y_test[:,0]-mean(y_test[:,0]),2))) )
        #fit2 = 1-math.sqrt(rmse2**2 / (sum((y_test[:,1]-mean(y_test[:,1]))**2)/Ntest) )
        #print('test fit = ', fit1, fit2, file=outfile)
        print('test fit = ', fit1, file=outfile)
        #print('test fit = ', fit1, fit2)
        
        # rmse1 = math.sqrt(sum((y_test_pred[:,0] - ypred_test[:,0])**2))/math.sqrt(Ntest)
        # rmse2 = math.sqrt(sum((y_test_pred[:,1] - ypred_test[:,1])**2))/math.sqrt(Ntest)
        # print('test rmse (prediction) = ', rmse1, rmse2, file=outfile)
        # print('test rmse (prediction) = ', rmse1, rmse2)
        # fit1 = 1-math.sqrt(rmse1**2 / (sum((y_test[:,0]-mean(y_test[:,0]))**2)/Ntest) )
        # fit2 = 1-math.sqrt(rmse2**2 / (sum((y_test[:,1]-mean(y_test[:,1]))**2)/Ntest) )
        # print('test fit = ', fit1, fit2, file=outfile)
        # print('test fit (prediction) = ', fit1, fit2)

        #print("Saving predicted sequence to file...")
        #np.savetxt("nnarx_test_"+str(num_layers)+"_"+str(num_neurons)+".csv", ysim_test, delimiter=",")
        #print("Done")
        
outfile.close()
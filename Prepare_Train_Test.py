from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import scipy
import numpy as np
import sys
import os
import glob
import time
import wfdb

class Prepare_Train_Test():
 def create_train_test(self, data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s, data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t):

    #Create train and test data

    #Shuffle data, labels and RR intervals from source and target domain
    x_s, z_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s = shuffle(data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s)
    x_t, z_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t= shuffle(data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t)

    inp_shape_1=x_s.shape[1]
    inp_shape_2=x_s.shape[2]

    #Split data into training and test data
    size_s=x_s.shape[0]
    split_90_10_s=int(size_s*0.9) #90% train, 10% test

    train_data_s=x_s[0:split_90_10_s]
    train_label_s=z_s[0:split_90_10_s]
    train_data_t=x_t[0:split_90_10_s]  
    train_label_t=z_t[0:split_90_10_s]
    test_data_s=x_s[split_90_10_s:size_s]
    test_label_s=z_s[split_90_10_s:size_s]
    test_data_t=x_t[split_90_10_s:size_s]
    test_label_t=z_t[split_90_10_s:size_s]

    train_data_RRinterval_s=data_RRinterval_s[0:split_90_10_s]
    train_data_prevRR_s=data_prevRR_s[0:split_90_10_s]
    train_data_prev_eight_RR_s=data_prev_eight_RR_s[0:split_90_10_s]
    train_data_RRinterval_t=data_RRinterval_t[0:split_90_10_s]
    train_data_prevRR_t=data_prevRR_t[0:split_90_10_s]
    train_data_prev_eight_RR_t=data_prev_eight_RR_t[0:split_90_10_s]

    test_RRinterval_s=data_RRinterval_s[split_90_10_s:size_s]
    test_prevRR_s=data_prevRR_s[split_90_10_s:size_s]
    test_prev_eightRR_s=data_prev_eight_RR_s[split_90_10_s:]
    test_RRinterval_t=data_RRinterval_t[split_90_10_s:size_s]
    test_prevRR_t=data_prevRR_t[split_90_10_s:size_s]
    test_prev_eightRR_t=data_prev_eight_RR_t[split_90_10_s:size_s]


    #Create training and test batches (Source and Target domain)

    batch_size=512 # Batch Size

    train_data_size=train_data_s.shape[0]
    test_data_size=test_data_s.shape[0]
    itr_len=int(train_data_size/batch_size)
    itr_len_test=int(test_data_size/batch_size)

    train_data_s=train_data_s[0:itr_len*batch_size]
    train_data_t=train_data_t[0:itr_len*batch_size]
    train_label_s=train_label_s[0:itr_len*batch_size]
    train_label_t=train_label_t[0:itr_len*batch_size]

    test_data_s=test_data_s[0:itr_len_test*batch_size]
    test_label_s=test_label_s[0:itr_len_test*batch_size]
    test_data_t=test_data_t[0:itr_len_test*batch_size]
    test_label_t=test_label_t[0:itr_len_test*batch_size]

    train_data_RRinterval_s=train_data_RRinterval_s[0:itr_len*batch_size]
    train_data_prevRR_s=train_data_prevRR_s[0:itr_len*batch_size]
    train_data_prev_eight_RR_s=train_data_prev_eight_RR_s[0:itr_len*batch_size]
    train_data_RRinterval_t=train_data_RRinterval_t[0:itr_len*batch_size]
    train_data_prevRR_t=train_data_prevRR_t[0:itr_len*batch_size]
    train_data_prev_eight_RR_t=train_data_prev_eight_RR_t[0:itr_len*batch_size]

    test_RRinterval_s=test_RRinterval_s[0:itr_len_test*batch_size]
    test_prevRR_s=test_prevRR_s[0:itr_len_test*batch_size]
    test_prev_eightRR_s=test_prev_eightRR_s[0:itr_len_test*batch_size]
    test_RRinterval_t=test_RRinterval_t[0:itr_len_test*batch_size]
    test_prevRR_t=test_prevRR_t[0:itr_len_test*batch_size]
    test_prev_eightRR_t=test_prev_eightRR_t[0:itr_len_test*batch_size]

    train_data_batch_s=arr = np.empty([itr_len, batch_size, train_data_s.shape[1], train_data_s.shape[2]])
    train_data_batch_t=arr = np.empty([itr_len, batch_size, train_data_t.shape[1], train_data_t.shape[2]])
    train_label_batch_s=arr = np.empty([itr_len, batch_size])
    train_label_batch_t=arr = np.empty([itr_len, batch_size])

    test_data_batch_s=arr = np.empty([itr_len_test, batch_size, test_data_s.shape[1], test_data_s.shape[2]])
    test_data_batch_t=arr = np.empty([itr_len_test, batch_size, test_data_t.shape[1], test_data_t.shape[2]])
    test_label_batch_s=arr = np.empty([itr_len_test, batch_size])
    test_label_batch_t=arr = np.empty([itr_len_test, batch_size])

    train_RRinterval_batch_s=arr = np.empty([itr_len, batch_size, train_data_RRinterval_s.shape[1]])
    train_prevRR_batch_s=arr = np.empty([itr_len, batch_size, train_data_prevRR_s.shape[1]])
    train_prev_eightRR_batch_s=arr = np.empty([itr_len, batch_size, train_data_prev_eight_RR_s.shape[1]])
    train_RRinterval_batch_t=arr = np.empty([itr_len, batch_size, train_data_RRinterval_t.shape[1]])
    train_prevRR_batch_t=arr = np.empty([itr_len, batch_size, train_data_prevRR_t.shape[1]])
    train_prev_eightRR_batch_t=arr = np.empty([itr_len, batch_size, train_data_prev_eight_RR_t.shape[1]])

    test_RRinterval_batch_s=arr = np.empty([itr_len_test, batch_size, test_RRinterval_s.shape[1]])
    test_prevRR_batch_s=arr = np.empty([itr_len_test, batch_size, test_prevRR_s.shape[1]])
    test_prev_eightRR_batch_s=arr = np.empty([itr_len_test, batch_size, test_prev_eightRR_s.shape[1]])
    test_RRinterval_batch_t=arr = np.empty([itr_len_test, batch_size, test_RRinterval_t.shape[1]])
    test_prevRR_batch_t=arr = np.empty([itr_len_test, batch_size, test_prevRR_t.shape[1]])
    test_prev_eightRR_batch_t=arr = np.empty([itr_len_test, batch_size, test_prev_eightRR_t.shape[1]])

    #Create training mini batches
    j=0
    for i in range(itr_len):
      train_data_batch_s[i]=train_data_s[j:j+batch_size]
      train_data_batch_t[i]=train_data_t[j:j+batch_size]
      train_label_batch_s[i]=train_label_s[j:j+batch_size]
      train_label_batch_t[i]=train_label_t[j:j+batch_size]

      train_RRinterval_batch_s[i]=train_data_RRinterval_s[j:j+batch_size]
      train_prevRR_batch_s[i]=train_data_prevRR_s[j:j+batch_size]
      train_prev_eightRR_batch_s[i]=train_data_prev_eight_RR_s[j:j+batch_size]
      train_RRinterval_batch_t[i]=train_data_RRinterval_t[j:j+batch_size]
      train_prevRR_batch_t[i]=train_data_prevRR_t[j:j+batch_size]
      train_prev_eightRR_batch_t[i]=train_data_prev_eight_RR_t[j:j+batch_size]
      j=j+batch_size

    #Create test mini batches
    j=0
    for i in range(itr_len_test):
      test_data_batch_s[i]=test_data_s[j:j+batch_size]
      test_data_batch_t[i]=test_data_t[j:j+batch_size]
      test_label_batch_s[i]=test_label_s[j:j+batch_size]
      test_label_batch_t[i]=test_label_t[j:j+batch_size]

      test_RRinterval_batch_s[i]=test_RRinterval_s[j:j+batch_size]
      test_prevRR_batch_s[i]=test_prevRR_s[j:j+batch_size]
      test_prev_eightRR_batch_s[i]=test_prev_eightRR_s[j:j+batch_size]
      test_RRinterval_batch_t[i]=test_RRinterval_t[j:j+batch_size]
      test_prevRR_batch_t[i]=test_prevRR_t[j:j+batch_size]
      test_prev_eightRR_batch_t[i]=test_prev_eightRR_t[j:j+batch_size]
      j=j+batch_size

    train_label_batch_s=train_label_batch_s.astype(int)
    train_label_batch_t=train_label_batch_t.astype(int)
    test_label_batch_s=test_label_batch_s.astype(int)
    test_label_batch_t=test_label_batch_t.astype(int)

    train_RRinterval_batch_s=train_RRinterval_batch_s.astype(int)
    train_prevRR_batch_s=train_prevRR_batch_s.astype(int)
    train_prev_eightRR_batch_s=train_prev_eightRR_batch_s.astype(int)
    train_RRinterval_batch_t=train_RRinterval_batch_t.astype(int)
    train_prevRR_batch_t=train_prevRR_batch_t.astype(int)
    train_prev_eightRR_batch_t=train_prev_eightRR_batch_t.astype(int)

    test_RRinterval_batch_s=test_RRinterval_batch_s.astype(int)
    test_prevRR_batch_s=test_prevRR_batch_s.astype(int)
    test_prev_eightRR_batch_s=test_prev_eightRR_batch_s.astype(int)
    test_RRinterval_batch_t=test_RRinterval_batch_t.astype(int)
    test_prevRR_batch_t=test_prevRR_batch_t.astype(int)
    test_prev_eightRR_batch_t=test_prev_eightRR_batch_t.astype(int)

    train_RRinterval_batch_s=np.reshape(train_RRinterval_batch_s,(train_RRinterval_batch_s.shape[0],train_RRinterval_batch_s.shape[1],-1))
    train_prevRR_batch_s=np.reshape(train_prevRR_batch_s,(train_prevRR_batch_s.shape[0],train_prevRR_batch_s.shape[1],-1))
    train_prev_eightRR_batch_s=np.reshape(train_prev_eightRR_batch_s,(train_prev_eightRR_batch_s.shape[0],train_prev_eightRR_batch_s.shape[1],-1))
    train_RRinterval_batch_t=np.reshape(train_RRinterval_batch_t,(train_RRinterval_batch_t.shape[0],train_RRinterval_batch_t.shape[1],-1))
    train_prevRR_batch_t=np.reshape(train_prevRR_batch_t,(train_prevRR_batch_t.shape[0],train_prevRR_batch_t.shape[1],-1))
    train_prev_eightRR_batch_t=np.reshape(train_prev_eightRR_batch_t,(train_prev_eightRR_batch_t.shape[0],train_prev_eightRR_batch_t.shape[1],-1))

    test_RRinterval_batch_s=np.reshape(test_RRinterval_batch_s,(test_RRinterval_batch_s.shape[0],test_RRinterval_batch_s.shape[1],-1))
    test_prevRR_batch_s=np.reshape(test_prevRR_batch_s,(test_prevRR_batch_s.shape[0],test_prevRR_batch_s.shape[1],-1))
    test_prev_eightRR_batch_s=np.reshape(test_prev_eightRR_batch_s,(test_prev_eightRR_batch_s.shape[0],test_prev_eightRR_batch_s.shape[1],-1))
    test_RRinterval_batch_t=np.reshape(test_RRinterval_batch_t,(test_RRinterval_batch_t.shape[0],test_RRinterval_batch_t.shape[1],-1))
    test_prevRR_batch_t=np.reshape(test_prevRR_batch_t,(test_prevRR_batch_t.shape[0],test_prevRR_batch_t.shape[1],-1))
    test_prev_eightRR_batch_t=np.reshape(test_prev_eightRR_batch_t,(test_prev_eightRR_batch_t.shape[0],test_prev_eightRR_batch_t.shape[1],-1))

    print('Training (source) shapes')
    print(train_data_batch_s.shape)
    print(train_RRinterval_batch_s.shape)
    print(train_prevRR_batch_s.shape)
    print(train_prev_eightRR_batch_s.shape)
    print(train_label_batch_s.shape)

    print('Training (target) shapes')
    print(train_data_batch_t.shape)
    print(train_RRinterval_batch_t.shape)
    print(train_prevRR_batch_t.shape)
    print(train_prev_eightRR_batch_t.shape)
    print(train_label_batch_t.shape)

    print('Test (target) shapes')
    print(test_data_batch_t.shape)
    print(test_RRinterval_batch_t.shape)
    print(test_prevRR_batch_t.shape)
    print(test_prev_eightRR_batch_t.shape)
    print(test_label_batch_t.shape)

    input_shape=(inp_shape_1,inp_shape_2)
    num_classes=4

    return train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, train_label_batch_t, test_data_batch_t, test_RRinterval_batch_t, test_prevRR_batch_t, test_prev_eightRR_batch_t, test_label_batch_t, input_shape, num_classes


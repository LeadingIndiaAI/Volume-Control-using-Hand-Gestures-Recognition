# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:09:33 2018

@author: SG1944
"""
import cv2
import os
import numpy as np
import pandas as pd
All_frame=[]
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/no'
i=0
k=0
framez1=[]
for file in os.listdir(base):
  
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0
   i=i+1
x1=len(All_frame)    
   
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/down'
i=0
k=0
framez1=[]
for file in os.listdir(base):
  
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0
   i=i+1
x2=len(All_frame)-x1     

base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/up'
i=0
k=0
framez1=[]
for file in os.listdir(base):
  
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0
   i=i+1
x3=len(All_frame)-x2-x1     

total_length=x2+x1+x3


label=np.ones((total_length,),dtype = int)
label[0:x1-1]= 0
label[x1:x1+x2-1] = 1
label[x1+x2:]=2

X_tr_array=np.array(All_frame)
train_data = [X_tr_array,label]
(train_set, y_train) = (train_data[0],train_data[1])
print(y_train.shape)

from keras.utils import np_utils, generic_utils




print('X_Train shape:', train_set.shape)
patch_size = 16
batch_size = 2
nb_classes = 3
nb_epoch =40
img_rows=132
img_cols=132
Y_train = np_utils.to_categorical(y_train, nb_classes)
print(Y_train.shape)
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

print(train_set.shape)

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,MaxPooling2D)
from collections import deque
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras
from keras.optimizers import SGD, RMSprop
from keras import optimizers
from keras.layers import Reshape
from keras.layers import LSTM
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
import cv2
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
from keras.models import model_from_json
from keras.callbacks import Callback
from keras import backend as K
input_shape=(16,132,132,3)
nb_classes=3

model = Sequential()

model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1),
            activation='relu', padding='same'), input_shape=input_shape))
model.add(TimeDistributed(Conv2D(64, (3,3),
            kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(256, (3,3),
           padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
       
model.add(TimeDistributed(Conv2D(512, (3,3),
          padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=False))

#model.add(Dropout(0.5))
model.add((Dense(1024, kernel_regularizer=regularizers.l2(0.01),activation='relu')))
##model.add(TimeDistributed(Dense(256, activation='relu')))
##model.add(TimeDistributed(Dense(128, activation='relu')))
#model.add(LSTM(4096, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(LSTM(512, return_sequences=False, dropout=0.5))
model.add(Dense(nb_classes, activation='softmax'))
sgd=SGD(lr=0.003)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for layer in model.layers:
 print(layer.get_output_at(0).get_shape().as_list())
print(model.summary())

model_json = model.to_json()
with open("timedistributed2.json", "w") as json_file:
    json_file.write(model_json)


X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.1, random_state=4)
print(y_train_new.shape)




def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    timedisloss=plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    print("plotting")
    timedisacc=plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    #plt.show()
    
    timedisloss.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/timedisloss2.png')
    timedisacc.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/timedisacc2.png')

    #c.savefig(r'C:\Users\sg1944\foo4.png')
    #plt.savefig('foo1.png')

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),batch_size=batch_size,verbose=1,nb_epoch = nb_epoch,shuffle=True)

visualizeHis(hist)
fname = r'/home/dgxuser103/Surbhi/9/Surbhi/data/timedistributed2.hdf5'
model.save_weights(fname,overwrite=True)












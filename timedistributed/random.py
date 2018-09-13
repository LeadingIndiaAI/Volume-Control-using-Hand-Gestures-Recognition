# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:11:52 2018

@author: SG1944
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:11:10 2018

@author: SG1944
"""

import keras
import cv2
import os
from keras.models import model_from_json
import numpy as np
json_file = open(r"C:\Users\sg1944\Surbhi\internship\model2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(r'C:\Users\sg1944\newweighttime.hdf5')
framez=[]
All_frames1=[1]
x=0
capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
while True:
    ret, frame = capture.read()
    if ret:
        x+=1
        
        gray=cv2.resize(frame,(132,132))
        framez.append(gray)
        print(x)
        if(x==16): 
          print('sss')
          frame2=np.array(framez)
          All_frames1[0]=frame2
          print(len(All_frames1))
          All_frames=np.array(All_frames1)
          print(All_frames.shape)
          setvid = All_frames.astype('float32')

          setvid -= np.mean(setvid)

          setvid /=np.max(setvid)
          
         # index = model.predict_classes(setvid)
          #prob_array = model.predict_proba(setvid)
          y_proba = model.predict(setvid)
          y_classes = y_proba.argmax(axis=-1)
          print(y_classes)
         # All_frames.insert(0,frame)
          
          #All_frames=np.array(All_frames)
          #results =loaded_model.predict(All_frames)
          All_frames = np.empty(All_frames.shape)

          framez=[]
          x=0
          
    cv2.imshow('frame', frame)
    cv2.waitKey(100) 
capture.release()
cv2.destroyAllWindows()


from keras.applications.inception_v3 import InceptionV3
import cv2
import os
from keras.models import model_from_json
import numpy as np
base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

from keras.models import Model
model=base_model
intermediate_layer_model = Model(inputs=base_model.input,
                                 outputs=base_model.get_layer('avg_pool').output)
#intermediate_output = intermediate_layer_model.predict(data)

def pred(img): 
    x=np.reshape(img,(1,299,299,3))
    intermediate_output = intermediate_layer_model.predict(x)
    ret_val=np.reshape(intermediate_output,(2048,))
    return ret_val
import numpy as np
import os
import pickle
sequence=[]
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/no'
i=0
k=0
framez1=[]
for file in os.listdir(base):
   gray=cv2.imread(os.path.join(base,file))
   img=cv2.resize(gray,(299,299))
   features=pred(img)
   framez1.append(features)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       sequence.append(ipt)
       framez1=[]
       k=0


with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data//No Ges.txt","wb") as fp:
  frame1=pickle.dump(sequence,fp)

sequence=[]
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/up'
i=0
k=0
framez1=[]
for file in os.listdir(base):
   gray=cv2.imread(os.path.join(base,file))
   img=cv2.resize(gray,(299,299))
   features=pred(img)
   framez1.append(features)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       sequence.append(ipt)
       framez1=[]
       k=0


with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data/swip up.txt","wb") as fp:
  frame1=pickle.dump(sequence,fp)

sequence=[]
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/down'
i=0
k=0
framez1=[]
for file in os.listdir(base):
   gray=cv2.imread(os.path.join(base,file))
   img=cv2.resize(gray,(299,299))
   features=pred(img)
   framez1.append(features)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       sequence.append(ipt)
       framez1=[]
       k=0


with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data/swip down.txt","wb") as fp:
  frame1=pickle.dump(sequence,fp)  


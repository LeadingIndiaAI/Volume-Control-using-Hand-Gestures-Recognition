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


json_file = open(r"C:\Users\Harinder\Documents\results\bothadam\inceptionmodel.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r'C:\Users\Harinder\Documents\results\bothadam\inceptionmodelweights.hdf5')
All_frames=[1]
framez=[]
x=0
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
while True:
    ret, frame = capture.read()
    if ret:
        x+=1
        img=cv2.resize(frame,(299,299))
        y=pred(img)
        framez.append(y)
        print(x)
        if(x==16): 
          #print('sss')
          frame2=np.array(framez)
          print(frame2.shape)
          All_frames[0]=frame2
          #print(len(All_frames1))
          All_frames=np.array(All_frames)
          #print(All_frames.shape)
          setvid = All_frames.astype('float32')

          setvid -= np.mean(setvid)

          setvid /=np.max(setvid)
          
          #pi=np.array(setvid)
          index = model.predict_classes(setvid)
          prob_array = model.predict_proba(setvid)
          print(prob_array)
          print(type(prob_array))
          x=prob_array.item(1)
          y=prob_array.item(2)
          print(x)
          print(y)
          if(x>y):
              print('up')
          if(x<y):
              print('down')
              
              
         # All_frames.insert(0,frame)
          
          #All_frames=np.array(All_frames)
          #results =loaded_model.predict(All_frames)
          #All_frames = np.empty(All_frames.shape)

          framez=[]
          x=0
          
    cv2.imshow('frame', frame)
    cv2.waitKey(1) 
capture.release()
cv2.destroyAllWindows()

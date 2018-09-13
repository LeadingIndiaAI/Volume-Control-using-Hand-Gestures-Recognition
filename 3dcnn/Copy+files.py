
# coding: utf-8

# In[9]:

import shutil
import os
from pandas import DataFrame
import pandas as pd
from shutil import copyfile


# In[10]:

df=pd.read_csv(r'E:\dataset lstm\gstrcsv.csv')

df = df[df.Gesture != 'Sliding Two Fingers Up']
df = df[df.Gesture != 'Sliding Two Fingers Down']
Folder=df['Folder']


# In[14]:

i=7586
x=0

for num in Folder:
    if num>7464:
        try:
         src = os.path.join(r'E:\test',str(num))
         src_files = os.listdir(src)
         for file_name in src_files:
            copyfile(os.path.join(src,file_name),os.path.join(r'E:\dataset lstm\Final\Final2\No gesture',str(i)+'.jpg'))
            i=i+1
        except:
            x=x+1
 #7465


# In[15]:

df=pd.read_csv(r'E:\dataset lstm\gstrcsv.csv')

df = df[df.Gesture != 'No gesture']
df = df[df.Gesture != 'Sliding Two Fingers Down']
Folder=df['Folder']


# In[16]:

i=7299
x=0
for num in Folder:
    if num>7464:
        try:
         src = os.path.join(r'E:\test',str(num))
         src_files = os.listdir(src)
         for file_name in src_files:
            copyfile(os.path.join(src,file_name),os.path.join(r'E:\dataset lstm\Final\Final2\Sliding Two Fingers Up',str(i)+'.jpg'))

            i=i+1
        except:
            x=x+1
             
        #7443


# In[17]:

df=pd.read_csv(r'E:\dataset lstm\gstrcsv.csv')

df = df[df.Gesture != 'No gesture']
df = df[df.Gesture != 'Sliding Two Fingers Up']
Folder=df['Folder']


# In[18]:

i=7601
x=0
for num in Folder:
    if num>7464:
      try:
        src = os.path.join(r'E:\test',str(num))
        src_files = os.listdir(src)
        for file_name in src_files:
            copyfile(os.path.join(src,file_name),os.path.join(r'E:\dataset lstm\Final\Final2\Sliding Two Fingers Down',str(i)+'.jpg'))

            i=i+1
      except:
        x=x+1
    #7424


# In[5]:

import os
import shutil
from shutil import copyfile

dir_src = r"E:\temp1"
dir_dst = r"E:\tem2"
for file in os.listdir(dir_src):

    src_file = os.path.join(dir_src, file)
    dst_file = os.path.join(dir_dst, file)
    copyfile(src_file, dst_file) 


# In[ ]:




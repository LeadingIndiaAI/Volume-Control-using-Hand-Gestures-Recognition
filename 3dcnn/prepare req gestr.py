# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 11:48:49 2018

@author: SG1944
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:21:47 2018

@author: Harinder
"""

import os
os.chdir(r'E:\dataset lstm')
import pandas as pd
import csv

x=pd.read_csv('jester-v1-train.csv')
y=x['34870;Drumming Fingers']
print(len(y))
number=[]
name=[]
for a in y:
    n=a.index(';')
    number.append(a[0:n])
    name.append(a[n+1:])
from pandas import DataFrame

df = DataFrame({'Gesture': name, 'Folder': number})





df = df[df.Gesture != 'Doing other things']
print(df)
df= df[df.Gesture != 'Drumming Fingers']
df=df[df.Gesture != 'Pulling Hand In']
df = df[df.Gesture != 'Pulling Two Fingers In']
df= df[df.Gesture != 'Pushing Hand Away']
df=df[df.Gesture != 'Pushing Two Fingers Away']
df = df[df.Gesture != 'Rolling Hand Backward']
df= df[df.Gesture != 'Rolling Hand Forward']
df=df[df.Gesture != 'Shaking Hand']
df = df[df.Gesture != 'Swiping Right']
df= df[df.Gesture != 'Sliding Two Fingers Left']
df=df[df.Gesture != 'Sliding Two Fingers Right']
df = df[df.Gesture != 'Swiping Left']
df= df[df.Gesture != 'Stop Sign']
df=df[df.Gesture != 'Swiping Down']
df = df[df.Gesture != 'Swiping Up']
df= df[df.Gesture != 'Thumb Down']
df=df[df.Gesture != 'Thumb Up']
df = df[df.Gesture != 'Turning Hand Clockwise']
df= df[df.Gesture != 'Turning Hand Counterclockwise']
df=df[df.Gesture != 'Zooming In With Full Hand']
df = df[df.Gesture != 'Zooming In With Two Fingers']
df= df[df.Gesture != 'Zooming Out With Full Hand']
df=df[df.Gesture != 'Zooming Out With Two Fingers']
print(df)
df.to_excel('test.xlsx', sheet_name='sheet1', index=False)


#Open .xlsx file in microsoft exel and save as gstrcsv.csv file


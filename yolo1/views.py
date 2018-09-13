from django.shortcuts import render

import sys
# Create your views here.
def request_page(request):
    return render(request,'base.html')

def formulate(request):
  import cv2
  capture = cv2.VideoCapture(0)
  if(request.GET.get('btn1') or request.GET.get('btn1') ):


      capture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
      capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
      while True:
          import os
          import math

          os.chdir(r'C:\Users\Harinder\Desktop\darkflow\darkflow-master')
          import cv2
          from darkflow.net.build import TFNet
          import numpy as np
          import time
          options = {
              'model': 'cfg/tiny-yolo-voc-1c.cfg',
              'load': 1531,
              'threshold': 0.11,

          }

          tfnet = TFNet(options)
          colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
          capture.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
          capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

          x = 0
          topl = []
          bottomr = []
          topl.insert(0, 0)
          bottomr.insert(0, 0)

          def vol(x2):
              from ctypes import cast, POINTER
              from comtypes import CLSCTX_ALL
              from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
              # global x1,y1
              # current centroid print(x2,y2)
              devices = AudioUtilities.GetSpeakers()
              interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
              volume = cast(interface, POINTER(IAudioEndpointVolume))
              curvol = volume.GetMasterVolumeLevel()
              # print('master',curvol)
              print(volume.GetVolumeRange())
              # dist = (math.hypot(x2-x1,y2-y1))
              # print('dist',curvol-dist,curvol+dist)
              if x2:
                  curvol = -1 * ((-1 * (curvol + 2)) % 65)
              elif curvol > -66:
                  curvol = -1 * ((-1 * (curvol - 2)) % 65)
              else:
                  curvol = 0
                  print("No further volume control")
              curvol = int(curvol)
              if curvol != None:
                  # print(curvol)
                  try:
                      volume.SetMasterVolumeLevel(curvol, None)
                  except:
                      print("exception")
              else:
                  print("no")

          while True:
              stime = time.time()
              ret, frame = capture.read()
              frame = cv2.flip(frame, 1)
              if ret:
                  x += 1
                  results = tfnet.return_predict(frame)
                  for color, result in zip(colors, results):
                      tl = (result['topleft']['x'], result['topleft']['y'])
                      br = (result['bottomright']['x'], result['bottomright']['y'])
                      print(tl)
                      if x % 2 == 0:

                          subtl = tl[0] - topl[0]
                          subbr = br[0] - bottomr[0]
                          print("Changed coordinate")
                          if subtl >= 0:
                              print("Left")

                          else:
                              print("Right")

                          print(subtl)
                          topl.insert(0, tl[0])
                          bottomr.insert(0, br[0])
                      label = result['label']
                      confidence = result['confidence']
                      text = '{}: {:.0f}%'.format(label, confidence * 100)
                      frame = cv2.rectangle(frame, tl, br, color, 5)
                      frame = cv2.putText(
                          frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                  cv2.imshow('frame', frame)
                  print('FPS {:.1f}'.format(1 / (time.time() - stime)))
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
      capture.release()
      cv2.destroyAllWindows()
  if (request.GET.get('btn2')):
      capture.release()
      cv2.destroyAllWindows()
  return render(request,'base.html')
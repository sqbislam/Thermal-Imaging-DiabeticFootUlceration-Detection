# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 18:53:47 2022

@author: Saqib
"""

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from math import floor
import cv2

def load_annotations(file):
  '''
    Takes a file and retruns annotations

    Input:
      file : path to csv file
    Output:
      annotations numpy array
  '''
  annot = []
  if not os.path.exists(file):
      assert "Error"
  with open(file, 'r') as f:
      reader = csv.reader(f)
      for idx, line in enumerate(reader):
          label = line[-1]
          # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
          line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
          label, x, y, name, w, h = list(line[:6])
          annot.append([x, y]) 
      return np.asarray(annot, dtype=np.float32)
  
def get_annotations(file):
    annotations = load_annotations(file)
    annotations_test = annotations.reshape(2,7,2) 
    return annotations_test


def draw_patches(image, annotations):
    BASE_ROI_SIZE = 5
    BASE_LARGE_ROI_SIZE = 8
    size = 1.3


    im = np.copy(image * 255)
  
    
    for idx, z in enumerate(annotations):
      half_patch = BASE_ROI_SIZE * size # pixels = 4x4 patches
      if(idx == 3 or idx == 6):
        half_patch = BASE_LARGE_ROI_SIZE * size # pixels = 4x4 patches
      x = z[0]
      y = z[1]
  
      if(idx == 3):
        topx = x + half_patch*2
        topy = y - half_patch*2
        
        botx = x 
        boty = y 
      else:
        topx = x - half_patch
        topy = y + half_patch
        
        botx = x + half_patch
        boty = y - half_patch
  
      top = (int(floor(topx)), int(floor(topy)))
      bot = (int(floor(botx)), int(floor(boty)))
  
      im = cv2.rectangle(im,top,bot,(255,0,0),1)
  
      # font
      font = cv2.FONT_HERSHEY_SIMPLEX
        
      # Blue color in BGR
      color = (255, 255, 255)
        
  
      # Using cv2.putText() method
      image = cv2.putText(im, f'{idx + 1}', top, font, 
                        0.4, color, 1, cv2.LINE_AA)
      plt.axis('off')
      plt.imshow(im, cmap="inferno")

    
    
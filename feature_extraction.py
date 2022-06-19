# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:49:47 2022

@author: Saqib
"""

from scipy import stats
import pywt
import pandas as pd
import pywt.data
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from  math import floor
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.BASE_ROI_SIZE = 5
        self.BASE_LARGE_ROI_SIZE = 8
        self.feat_dict = self.generate_feat_dict(7)
    
    def extract_patch_and_save(self, data, annotations, type_name, dir="", scale=1.0, orient="L"):
  
        roi_patches = []
        
        # Repeat process for seven ROIS. 
        for roi_idx in range(0,7):
          im = np.copy(data * 255)
        
          # Extract ROI
          ROI = annotations[roi_idx]
          half_patch = self.BASE_ROI_SIZE * scale  # Joint region smaller ROIs
          if(orient == "L"):
            if(roi_idx == 3 or roi_idx == 6):
              half_patch = self.BASE_LARGE_ROI_SIZE * scale # Plantar region large ROIs
          elif(orient == "R"): 
            if(roi_idx >= 5):
              half_patch = self.BASE_LARGE_ROI_SIZE * scale # pixels = 4x4 patches
        
        
          x = ROI[0] 
          y = ROI[1]
        
          
          
          botx = x 
          boty = y 
            
          topx = int(floor(x - half_patch))
          topy = int(floor(y - half_patch))
        
          botx = int(floor(x + half_patch))
          boty = int(floor(y + half_patch))
        
        
          if(orient == "L"):
            # Make ROI 6 and 7 larger
            if(roi_idx == 3 or roi_idx == 6):
                topy = int(floor(y - half_patch * 2))
                boty = int(floor(y))
          else:
            # Make ROI 6 and 7 larger
              if(roi_idx > 5):
                topy = int(floor(y - half_patch * 2))
                boty = int(floor(y))
        
        
        
          top = (topx, topy)
          bot = (botx, boty)
        
          ROI_slice = im[topy + 1 : boty, topx + 1: botx]
        
          roi_patches.append(ROI_slice)
       
            # save_image(ROI_slice, name, dir)
            # print(ROI_slice)
            # im = cv2.rectangle(im,top,bot,(255,0,0),1)
            # plt.figure()
            # plt.imshow(im)
        
        return roi_patches
     
    def extract_glcm_feat(self, data, feat_dict, idx):
    
      patch_normed = data
      patch = patch_normed.astype(np.uint8)
      glcm = greycomatrix(patch, [2], [0], symmetric=True, normed=True)
      feat_dict['dissimilarity-' + str(idx)].append(greycoprops(glcm, 'dissimilarity')[0, 0])
      feat_dict['contrast-' + str(idx)].append(greycoprops(glcm, 'contrast')[0, 0])
      feat_dict['homogeneity-' + str(idx)].append(greycoprops(glcm, 'homogeneity')[0, 0])
      feat_dict['ASM-' + str(idx)].append(greycoprops(glcm, 'ASM')[0, 0])
      feat_dict['energy-' + str(idx)].append(greycoprops(glcm, 'energy')[0, 0])
      feat_dict['correlation-' + str(idx)].append(greycoprops(glcm, 'correlation')[0, 0])
      feat_dict['mean-' + str(idx)].append(np.mean(data))
      return feat_dict

    def extract_image_stats(self, im, feat_dict, idx):
      st = stats.describe(im.flatten())
      feat_dict['skewness-' + str(idx)].append(st.skewness)
      feat_dict['kurtosis-' + str(idx)].append(st.kurtosis)
      feat_dict['variance-' + str(idx)].append(st.variance)
      return feat_dict

        
    def generate_features(self, image, annotations):
        arr = self.extract_patch_and_save(image, annotations, type_name="DM", dir = "", orient="L", scale=1.4)
        for i,roi in enumerate(arr):
            self.extract_glcm_feat(roi, self.feat_dict, i+1)
            self.extract_image_stats(roi, self.feat_dict, i+1)
        dataDL = pd.DataFrame.from_dict(self.feat_dict)
        return dataDL
        
        
    def generate_feat_dict(self, roi_number):
      feat_arr = [ "contrast", "ASM", "correlation", "mean", "dissimilarity","homogeneity","energy","skewness","kurtosis","variance"]
      feat_dict = {}
      # Create empty array for each feat
      for name in feat_arr:
        for i in range(1, roi_number + 1):
          feat_dict[name + "-" + str(i)] = []
        
      return feat_dict



# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:19:52 2019

@author: tiger
"""

""" Compute accuracy of segmentations between 2 different segmenters 

     loads in files 2 by 2
     binarizes and computes cc
     finds out how many are identical
     compute accuracy, AUC(?) ect...
     
"""

from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os
import zipfile
import scipy

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *
import glob, os


input_path = 'E:/3) Optic nerve project/compare Barbara v Etienne input/'


""" FOR UNET OUTPUT ==> must im_open + remove small objects """


""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*.tif'))
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]


all_PPV = []
all_F1 = []
all_sens = []
all_blebs_ground = 0
all_blebs_compare = 0
thresh_min_bleb_size = 0
for i in range(0, len(examples), 2):
     
     input_name_1 = examples[i + 1]['input']
     input_1 = np.asarray(Image.open(input_name_1), dtype=np.float32)
     input_1[input_1 > 0] = 1
     
     
     input_name_2 = examples[i ]['input']
     input_2 = np.asarray(Image.open(input_name_2), dtype=np.float32)  
     input_2[input_2 > 0] = 1
     
     overlap_im = input_1 + input_2
    
     overlap_bin = overlap_im > 0
     labelled = measure.label(overlap_bin)
     cc_overlap = measure.regionprops(labelled, intensity_image=overlap_im)
         
     """ (1) Find # True Positives identified (overlapped) """
     masked = np.zeros(input_1.shape)
     all_no_overlap = np.zeros(input_1.shape)
     truth_no_overlap = np.zeros(input_1.shape)   # ALL False Negatives
     seg_no_overlap = np.zeros(input_1.shape)     # All False Positives
     for M in range(len(cc_overlap)):
          overlap_val = cc_overlap[M]['MaxIntensity']
          overlap_coords = cc_overlap[M]['coords']
          if overlap_val > 1 and len(overlap_coords) > thresh_min_bleb_size:    # if there is overlap
              for T in range(len(overlap_coords)):
                  masked[overlap_coords[T,0], overlap_coords[T,1]] = overlap_bin[overlap_coords[T,0], overlap_coords[T,1]]   # TRUE POSITIVES
          else:  # no overlap
              for T in range(len(overlap_coords)):
                  all_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_bin[overlap_coords[T,0], overlap_coords[T,1]]     
                  #truth_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
                  #seg_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
         
     """ find false positives """
     false_p = all_no_overlap + input_2  # blebs that are identified by user #2 but do NOT show up in input_1
     labelled = measure.label(input_2)
     cc_overlap = measure.regionprops(labelled, intensity_image=false_p)
     for M in range(len(cc_overlap)):
          overlap_val = cc_overlap[M]['MaxIntensity']
          overlap_coords = cc_overlap[M]['coords']
          if overlap_val > 1 and len(overlap_coords) > thresh_min_bleb_size:    # if there is overlap
              for T in range(len(overlap_coords)):
                  seg_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = false_p[overlap_coords[T,0], overlap_coords[T,1]]
     
     
     """ find false negatives """
     false_n = all_no_overlap + input_1  # blebs that are identified by user #1 but do NOT show up in input_2
     labelled = measure.label(input_1)
     cc_overlap = measure.regionprops(labelled, intensity_image=false_n)
     for M in range(len(cc_overlap)):
          overlap_val = cc_overlap[M]['MaxIntensity']
          overlap_coords = cc_overlap[M]['coords']
          if overlap_val > 1 and len(overlap_coords) > thresh_min_bleb_size:    # if there is overlap
              for T in range(len(overlap_coords)):
                  truth_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = false_n[overlap_coords[T,0], overlap_coords[T,1]]
          
       
     labelled = measure.label(masked)
     cc_masked_TPs = measure.regionprops(labelled)
     TP = len(cc_masked_TPs)
          
     labelled = measure.label(truth_no_overlap)
     cc_truth_FNs = measure.regionprops(labelled)
     FN = len(cc_truth_FNs)
         
     labelled = measure.label(seg_no_overlap)
     cc_seg_FPs = measure.regionprops(labelled)
     FP = len(cc_seg_FPs)
             
     #accuracy = (TP + TN)/(TP + TN + FP + FN)
     PPV = TP/(TP + FP)
     F1_score = (2*TP)/(2*TP + FP + FN)
     sensitivity = TP/(TP + FN)
     print("PPV value for image %d is: %.3f" %(i + 1, PPV))    
     print("sens value for image %d is: %.3f" %(i + 1, sensitivity))        
     print("F1_score value for image %d is: %.3f\n" %(i + 1, F1_score))
     
     all_PPV.append(PPV)
     all_F1.append(F1_score)
     all_sens.append(sensitivity)
     all_blebs_ground = all_blebs_ground + TP + FN
     all_blebs_compare = all_blebs_compare + TP + FP


print("Mean PPV value is: %.3f" %(sum(all_PPV)/len(all_PPV)))
print("Mean sens value is: %.3f" %(sum(all_sens)/len(all_sens)))
print("Mean F1 value is: %.3f" %(sum(all_F1)/len(all_F1)))
print("Overall # of GROUND blebs counted is: %.3f" %(all_blebs_ground))
print("Overall # of COMPARE blebs counted is: %.3f" %(all_blebs_compare))


""" Over 8 slices:
Etienne as ground v Barbara
Mean PPV value is: 0.763
Mean sens value is: 0.498
Mean F1 value is: 0.566

Etienne as ground v UNet
Mean PPV value is: 0.676
Mean sens value is: 0.726
Mean F1 value is: 0.692
Overall # of GROUND blebs counted is: 173.000
Overall # of COMPARE blebs counted is: 190.000

Etienne v. Fiji
Mean PPV value is: 0.279
Mean sens value is: 0.264
Mean F1 value is: 0.267
Overall # of GROUND blebs counted is: 208.000
Overall # of COMPARE blebs counted is: 201.000


Barbara as ground v Etienne
Mean PPV value is: 0.498
Mean sens value is: 0.763
Mean F1 value is: 0.566

Barbara as ground v UNet
Mean PPV value is: 0.535
Mean sens value is: 0.803
Mean F1 value is: 0.611
Overall # of GROUND blebs counted is: 135.000
Overall # of COMPARE blebs counted is: 196.000


"""
            
plt.figure(); plt.imshow(input_1); plt.title("Input 1")
plt.figure(); plt.imshow(input_2); plt.title("Input 1")

plt.figure(); plt.imshow(masked); plt.title("True positives")
plt.figure(); plt.imshow(truth_no_overlap); plt.title("False Negatives")
plt.figure(); plt.imshow(seg_no_overlap); plt.title("False Positives")
     




















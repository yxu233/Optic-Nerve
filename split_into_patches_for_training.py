
# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???

@author: Tiger
"""
import tensorflow as tf
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
import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from data_functions import *
from split_im_to_patches import *
from UNet import *
import glob, os


import tkinter
from tkinter import filedialog
import os
    

"""  Currently assumes:
    
    R ==> 
    G ==> Nanofibers
    B ==> 
    dot ==> DAPI mask
    
"""


root = tkinter.Tk()
input_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Checkpoints/",
                                        title='Please select checkpoint directory')
input_path = input_path + '/'

#sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
#                                        title='Please select saving directory')
#sav_dir = sav_dir + '/'

""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*.ome.tif'))
examples = [dict(input=i,truth=i.replace('.ome.tif','_N_truth.tif')) for i in images]

cleaned = 0
uncleaned = 0
tf_size = 1024
for i in range(len(images)):

#for i in range(1):
    
    input_name = examples[i]['input']
    
    # NORMALIZED BECAUSE IMAGE IS uint16 ==> do same when actually running images!!!
    input_im = np.asarray(Image.open(input_name))
    if input_im.dtype == 'uint16':
        input_im = np.asarray(input_im, dtype=np.float32)
        input_im = cv.normalize(input_im, 0, 255, cv.NORM_MINMAX)
        
    input_im = np.asarray(input_im, dtype= np.float32)

    truth_name = examples[i]['truth']
    truth_tmp = np.asarray(Image.open(truth_name), dtype=np.float32)
    


    # make better names for later saving
    input_name = input_name.split('.')[0];
    truth_name = truth_name.split('_truth')[0];

    
    patches = patchify(input_im, patch_shape=(tf_size,tf_size), overlap=500)
    if not type(patches) is np.ndarray:
        patches = np.array(patches)
        
    patches_truth = patchify(truth_tmp, patch_shape=(tf_size,tf_size), overlap=500)
    if not type(patches_truth) is np.ndarray:
        patches_truth = np.array(patches_truth)

    """ if do split patches, then:
                (1) need to skip any patches that are empty to save time
                (2) analyze each patch image individually, then recombine
    """
    if patches.any():
        seg_output_patches = np.zeros(np.shape(patches))
        idx = 0
        for idx in range(len(patches)): 
                  
                    pos_or_neg = '';
                    
                    input_im = patches[idx]
                    truth_im = patches_truth[idx]
                                        
                    input_RGB = np.zeros(np.shape(input_im) + (3,))
                    input_RGB[:, :, 1] = input_im * 255
                    input_RGB = np.asarray(input_RGB, dtype=np.uint8)
                    input_im = input_RGB
                    #input_im[input_im < 5] = 0

                    # save as "neg" if nothing in image
                    if np.count_nonzero(truth_im) == 0:
                        #seg_train = np.zeros(np.shape(patches[0])) 
                        #seg_train[0:100, 0:500] = 25;   # for debugging
                        #seg_output_patches[idx, :, :] = seg_train
                        #idx = idx + 1     
                        print("skipped")
                        pos_or_neg = '_neg'
                        #continue
                    else:
                        pos_or_neg = '_pos'

                    input_im = Image.fromarray(np.asarray(input_im, dtype=np.uint8))
                    truth_im = Image.fromarray(np.asarray(truth_im, dtype=np.uint8))
                    input_im.save(input_name + '_N_patch_' + "%d" % (idx,) + pos_or_neg + '_input.tif')
                    truth_im.save(truth_name + '_patch_'  + "%d" % (idx,) + pos_or_neg + '_truth.tif')
    
    uncleaned = uncleaned + 1
    
    print(idx)
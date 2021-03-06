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

sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                        title='Please select saving directory')
sav_dir = sav_dir + '/'

""" Load filenames from zip """
onlyfile_names = read_file_names(input_path)

counter = list(range(len(onlyfile_names)))  # create a counter, so can randomize it
counter = np.array(counter)


cleaned = 0
uncleaned = 0
for i in range(len(onlyfile_names)):
    
    """ read in input images and corresponding masks """
    input_im = readIm_counter(input_path, onlyfile_names, fileNum = i)
    
    """ delete all slices that don't have a mask """
    
    
    
    """ save all remaining slices as single channel .tff """
    
    # IF TOO BIG, THEN MIGHT NEED TO SPLIT MORE INTO SMALLER SCENES ==> adaptcrop
    
    
    """ save all masks as single channel .tiff with 255 == truth """
    
    
    filename = onlyfile_names[counter[i]]
    input_im, truth_im = load_training_ZIP(myzip, filename)
    
    filename = filename.split('/')[-1]
   
    pos = '_neg'
    if truth_im[:, :, 1].any():
        pos = '_pos'
        print(i)        
    
    
    # if has nanofibers
    if input_im.shape[-1] > 3:
        slice_num = 3
    else:
        slice_num = 1
        
    
    # Clean small body sizes
    if np.count_nonzero(input_im[:, :, slice_num]) < minDAPIsize:
        cleaned = cleaned + 1
        print('Cleaned: ' + str(cleaned))
        continue
    
    # deal with the fiber channe;
    if slice_num == 3 and input_im[:, :, 1].any():
        nanofibers = input_im[:, :, 1]
        nanofibers = Image.fromarray(input_im[:, :, 1].astype('uint8'))
        nanofibers.save(sav_dir + 'myelin_' + filename + '_' + "%07d" % (uncleaned,) + pos + '_NANOFIBERS.tif')
        
        input_im[:, :, 1] = input_im[:, :, 3]
        input_im = input_im[:, :, 0:3]
    

    input_im = Image.fromarray(input_im.astype('uint8'))
    truth_im = Image.fromarray((truth_im[:,:,1] * 255).astype('uint8'))
    
    input_im.save(sav_dir + 'myelin_' + filename + '_' + "%07d" % (uncleaned,) + pos + '_input.tif')
    truth_im.save(sav_dir + 'myelin_' + filename + '_'  + "%07d" % (uncleaned,) + pos + '_truth.tif')
    
    uncleaned = uncleaned + 1
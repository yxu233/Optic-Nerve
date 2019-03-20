# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

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

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *
import glob, os


import tkinter
from tkinter import filedialog
import os
    

truth = 0


root = tkinter.Tk()
sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                        title='Please select saving directory')
sav_dir = sav_dir + '/'

# Initialize everything with specific random seeds for repeatability
tf.reset_default_graph() 
tf.set_random_seed(1); np.random.seed(1)

"""  Network Begins:
"""
## for saving
#s_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Checkpoints/3rd_run_SHOWCASE/'
s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/2nd_OPTIC_NERVE_run_full_dataset/'
#s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/3rd_OPTIC_NERVE_large_network/'

## for input
#input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/'
input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.16/EAE_A3/'
#input_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Training Data/'


""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
examples = [dict(input=i,truth=i.replace('.tif','truth.tif')) for i in images]

#examples = [dict(input=i.replace('_truth.tif', '.tif'),truth=i) for i in images]


# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 1024, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name = 'weighted_labels')


""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network_SMALL(x, y_, training)
#y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()
#num_check = 279000
num_check = 45000
checkpoint = '_' + str(num_check)
saver.restore(sess, s_path + 'check' + checkpoint)
    

# Required to initialize all
batch_size = 1;

batch_x = []; batch_y = [];
weights = [];

plot_jaccard = [];

output_stack = [];
output_stack_masked = [];
for i in range(len(examples)):
        
        input_name = examples[i]['input']
        input_im = np.asarray(Image.open(input_name), dtype=np.float32)
        input_save = np.copy(input_im)
         
        """ maybe remove normalization??? """
        input_im = normalize_im(input_im, mean_arr, std_arr) 
 

        """ if trying to run with test images and not new images, load the truth """
        if truth:
            truth_name = examples[i]['truth']
            truth_tmp = np.asarray(Image.open(truth_name), dtype=np.float32)
                   
            """ convert truth to 2 channel image """
            if "_neg_" in truth_name:
                truth_im = np.zeros(np.shape(truth_tmp) + (2,))
                truth_im[:, :, 0] = np.ones(np.shape(truth_tmp))   # background
                truth_im[:, :, 1] = np.zeros(np.shape(truth_tmp))   # blebs
            else:
                channel_1 = np.copy(truth_tmp)
                channel_1[channel_1 == 0] = 1
                channel_1[channel_1 == 255] = 0
                        
                channel_2 = np.copy(truth_tmp)
                channel_2[channel_2 == 255] = 1   
                
                truth_im = np.zeros(np.shape(truth_tmp) + (2,))
                truth_im[:, :, 0] = channel_2   # background
                truth_im[:, :, 1] = channel_1   # blebs
            
            blebs_label = np.copy(truth_im[:, :, 1])

            """ Get spatial AND class weighting mask for truth_im """
            sp_weighted_labels = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
            
            """ OR DO class weighting ONLY """
            #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
            
            """ Create a matrix of weighted labels """
            weighted_labels = np.copy(truth_im)
            weighted_labels[:, :, 1] = sp_weighted_labels

            batch_y.append(truth_im)
            weights.append(weighted_labels)
            
        else:
            batch_y.append(np.zeros([1024,1024,2]))
            weights.append(np.zeros([1024,1024,2]))

            
        """ set inputs and truth """
        batch_x.append(input_im)

                
        """ Plot for debug """
        if truth:
            plt.figure(1); 
            plt.subplot(221); plt.imshow(np.asarray(input_im, dtype = np.uint8)); plt.title('Input');
            plt.subplot(222); plt.imshow(sp_weighted_labels); plt.title('weighted');    plt.pause(0.005)
            plt.subplot(223); plt.imshow(channel_1); plt.title('background');
            plt.subplot(224); plt.imshow(channel_2); plt.title('blebs');

    
        """ Feed into training loop """
        feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 
        print('Analyzed: %d of total: %d' %(i + 1, len(examples)))
           
           
        plt.close(2)
       
        """ Training Jaccard """
        jacc_t = jaccard.eval(feed_dict=feed_dict)
        plot_jaccard.append(jacc_t)           
                              
        """ Plot for debug """
        feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
        output_train = softMaxed.eval(feed_dict=feed_dict)
        seg_train = np.argmax(output_train, axis = -1)[0]              
                        
        #plt.figure(2);
        plt.figure(num=2, figsize=(40, 40), dpi=80, facecolor='w', edgecolor='k')
        if truth:  # plot truth
            plt.subplot(121); plt.imshow(truth_tmp); plt.title('Truth');
        else:  # or just plot the input image
            plt.subplot(121); plt.imshow(np.asarray(input_save, dtype=np.uint8)); plt.title('Input');
            
        filename_split = input_name.split('\\')[-1]
        filename_split = filename_split.split('.')[0]
            
        plt.subplot(122); plt.imshow(seg_train); plt.title('Output');                            
        plt.savefig(sav_dir + filename_split + '_' + str(i) + '_compare_output.png', bbox_inches='tight')
              
        batch_x = []; batch_y = []; weights = [];
              
        plt.imsave(sav_dir + filename_split + '_' + str(i) + '_output_mask.tif', (seg_train), cmap='binary_r')
        
        """ Post processing """

        # (1) Mask the raw image with the seg output
        input_cp = np.copy(input_save)
        input_cp[seg_train == 0] = 0
        
        # convert to SINGLE channel + uint8, then apply threshold
        input_cp = np.asarray(input_cp[:, :, 1], dtype=np.uint8)
        th2 = cv.adaptiveThreshold(input_cp,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
        #ret,thresh1 = cv.threshold(input_cp,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # use thresholded mask to set things to zero in the original seg output
        seg_train_masked = np.copy(seg_train)
        seg_train_masked[th2 == 0] = 0
        
        # clean by eroding then dilating
        #kernel = np.ones((5,5),np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        seg_train_dil = cv.morphologyEx(np.asarray(seg_train_masked, dtype=np.uint8), cv.MORPH_OPEN, kernel)
        plt.imsave(sav_dir + filename_split + '_' + str(i) + '_output_mask_MASKED.tif', (seg_train_dil), cmap='binary_r')



        """ Save as 3D stack """
        if len(output_stack) == 0:
            output_stack = seg_train
            output_stack_masked = seg_train_masked
        else:
            output_stack = np.dstack([output_stack, seg_train])
            output_stack_masked = np.dstack([output_stack_masked, seg_train_masked])


""" Post-processing """

# 1) delete anything that "migrates" between frames (i.e. identified a sheath by accident) ==> then use to train on 3D images???
# 2) use outputs as masks over top of the raw image. Then BINARIZE to get a better segmentation of the output image!!!
#        ==> may need to do this b/c the ground truth is over-segmented!!!
#        ***can use parameters # 1 and 2 to clean images and then used cleaned images to re-train 3D model!!!











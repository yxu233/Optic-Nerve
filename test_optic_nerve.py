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

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *
import glob, os



truth = 1

# Initialize everything with specific random seeds for repeatability
tf.reset_default_graph() 
tf.set_random_seed(1); np.random.seed(1)

"""  Network Begins:
"""
# for saving
s_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Checkpoints/test/'
# for input
input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Training Data/'

""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]

#examples = [dict(input=i.replace('_truth.tif', '.tif'),truth=i) for i in images]


# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 1024, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name = 'weighted_labels')


""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network_SMALL(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()
num_check = 20000
checkpoint = '_' + str(num_check)
saver.restore(sess, s_path + 'check' + checkpoint)
    

# Required to initialize all
batch_size = 1; 

batch_x = []; batch_y = [];
weights = [];

plot_jaccard = [];

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
            channel_1 = np.copy(truth_tmp)
            channel_1[channel_1 == 0] = 1
            channel_1[channel_1 == 255] = 0
                
            channel_2 = np.copy(truth_tmp)
            channel_2[channel_2 == 255] = 1   
 
            truth_im = np.zeros(np.shape(truth_tmp) + (2,))
            truth_im[:, :, 0] = channel_1   # background
            truth_im[:, :, 1] = channel_2   # blebs
            
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
        plt.figure(1); 
        plt.subplot(221); plt.imshow(np.asarray(input_im, dtype = np.uint8)); plt.title('Input');
        plt.subplot(222); plt.imshow(sp_weighted_labels); plt.title('weighted');    plt.pause(0.005)
        plt.subplot(223); plt.imshow(channel_1); plt.title('background');
        plt.subplot(224); plt.imshow(channel_2); plt.title('blebs');

    
        """ Feed into training loop """
        feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 
        print('Analyzed: %d of total: %d' %(i, len(examples)))
           
           
        plt.close(2)
       
        """ Training Jaccard """
        jacc_t = jaccard.eval(feed_dict=feed_dict)
        plot_jaccard.append(jacc_t)           
                              
        """ Plot for debug """
        feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
        output_train = softMaxed.eval(feed_dict=feed_dict)
        seg_train = np.argmax(output_train, axis = -1)[0]              
                        
        plt.figure(2);
        if truth:  # plot truth
            plt.subplot(121); plt.imshow(truth_tmp); plt.title('Truth');
        else:  # or just plot the input image
            plt.subplot(121); plt.imshow(np.asarray(input_save, dtype=np.uint8)); plt.title('Input');
            
        plt.subplot(122); plt.imshow(seg_train); plt.title('Output');                            
        plt.savefig(s_path + '_' + str(i) + '_output.png')
              
        batch_x = []; batch_y = []; weights = [];
              

from __future__ import print_function

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
    


""" Convert voxel list to array """
def convert_vox_to_matrix(voxel_idx, zero_matrix):
    for row in voxel_idx:
        #print(row)
        zero_matrix[(row[0], row[1], row[2])] = 1
    return zero_matrix

""" For plotting the output as an interactive scroller"""
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


""" Only keeps objects in stack that are 5 slices thick!!!"""
def slice_thresh(output_stack, slice_size=5):
    binary_overlap = output_stack > 0
    labelled = measure.label(binary_overlap)
    cc_overlap = measure.regionprops(labelled)
    
    all_voxels = []
    all_voxels_kept = []; total_blebs_kept = 0
    all_voxels_elim = []; total_blebs_elim = 0
    total_blebs_counted = len(cc_overlap)
    for bleb in cc_overlap:
        cur_bleb_coords = bleb['coords']
    
        # get only z-axis dimensions
        z_axis_span = cur_bleb_coords[:, -1]
    
        min_slice = min(z_axis_span)
        max_slice = max(z_axis_span)
        span = max_slice - min_slice
    
        """ ONLY KEEP OBJECTS that span > 5 slices """
        if span >= slice_size:
            print("WIDE ENOUGH object") 
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = cur_bleb_coords
            else:
                all_voxels_kept = np.append(all_voxels_kept, cur_bleb_coords, axis = 0)
                
            total_blebs_kept = total_blebs_kept + 1
        else:
            print("NOT wide enough")
            if len(all_voxels_elim) == 0:
                print("came here")
                all_voxels_elim = cur_bleb_coords
            else:
                all_voxels_elim = np.append(all_voxels_elim, cur_bleb_coords, axis = 0)
                
            total_blebs_elim = total_blebs_elim + 1
       
        if len(all_voxels) == 0:   # if it's empty, initialize
            all_voxels = cur_bleb_coords
        else:
            all_voxels = np.append(all_voxels, cur_bleb_coords, axis = 0)
            
    print("Total kept: " + str(total_blebs_kept) + " Total eliminated: " + str(total_blebs_elim))
    
    
    """ convert voxels to matrix """
    all_seg = convert_vox_to_matrix(all_voxels, np.zeros(output_stack.shape))
    all_blebs = convert_vox_to_matrix(all_voxels_kept, np.zeros(output_stack.shape))
    all_eliminated = convert_vox_to_matrix(all_voxels_elim, np.zeros(output_stack.shape))
    
    return all_seg, all_blebs, all_eliminated


""" Find vectors of movement and eliminate blobs that migrate """
def distance_thresh(all_blebs_THRESH, average_thresh=15, max_thresh=15):
    
    # (1) Find and plot centroid of each 2D image object:
    centroid_matrix_3D = np.zeros(np.shape(all_blebs_THRESH))
    for i in range(len(all_blebs_THRESH[0, 0, :])):
        bin_cur_slice = all_blebs_THRESH[:, :, i] > 0
        label_cur_slice = measure.label(bin_cur_slice)
        cc_overlap_cur = measure.regionprops(label_cur_slice)
        
        for obj in cc_overlap_cur:
            centroid_matrix_3D[(int(obj['centroid'][0]),) + (int(obj['centroid'][1]),) + (i,)] = 1   # the "i" puts the centroid in the correct slice!!!
        
        #print(i)
        
    # (2) use 3D cc_overlap to find clusters of centroids
    binary_overlap = all_blebs_THRESH > 0
    labelled = measure.label(binary_overlap)
    cc_overlap_3D = measure.regionprops(labelled)
        
    all_voxels_kept = []; num_kept = 0
    all_voxels_elim = []; num_elim = 0
    for obj3D in cc_overlap_3D:
        
        slice_idx = np.unique(obj3D['coords'][:, -1])
        
        cropped_centroid_matrix = centroid_matrix_3D[:, :, min(slice_idx) : max(slice_idx) + 1]
        
        mask = np.ones(np.shape(cropped_centroid_matrix))

        translate_z_coords = obj3D['coords'][:, 0:2]
        z_coords = obj3D['coords'][:, 2:3]  % min(slice_idx)   # TRANSLATES z-coords to 0 by taking modulo of smallest slice index!!!
        translate_z_coords = np.append(translate_z_coords, z_coords, -1)
        
        obj_mask = convert_vox_to_matrix(translate_z_coords, np.zeros(cropped_centroid_matrix.shape))
        mask[obj_mask == 1] = 0 

        tmp_centroids = np.copy(cropped_centroid_matrix)  # contains only centroids that are masked by array above
        tmp_centroids[mask == 1] = 0
        
        
        ##mask = np.ones(np.shape(centroid_matrix_3D))
        ##obj_mask = convert_vox_to_matrix(obj3D['coords'], np.zeros(output_stack.shape))
        ##mask[obj_mask == 1] = 0 
    
        ##tmp_centroids = np.copy(centroid_matrix_3D)  # contains only centroids that are masked by array above
        ##tmp_centroids[mask == 1] = 0
        
        cc_overlap_cur_cent = measure.regionprops(np.asarray(tmp_centroids, dtype=np.int))  
        
        list_centroids = []
        for centroid in cc_overlap_cur_cent:
            if len(list_centroids) == 0:
                list_centroids = centroid['coords']
            else:
                list_centroids = np.append(list_centroids, centroid['coords'], axis = 0)
    
        sorted_centroids = sorted(list_centroids,key=lambda x: x[2])  # sort by column 3
        
        
        """ Any object with only 1 or less centroids is considered BAD, and is eliminated"""
        if len(sorted_centroids) <= 1:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            continue;
        
    
        # (3) Find distance from 1st - 2nd - 3rd - 4th - 5th ect... centroids
        all_distances = []
        for i in range(len(sorted_centroids) - 1):
            center_1 = sorted_centroids[i]
            center_2 = sorted_centroids[i + 1]
            
            # Find distance:
            dist = math.sqrt(sum((center_1 - center_2)**2))           # DISTANCE FORMULA
            #print(dist)
            all_distances.append(dist)
        average_dist = sum(all_distances)/len(all_distances)
        max_dist = max(all_distances)
        
        
        # (4) If average distance is BELOW thresdhold, then keep the 3D cell body!!!
        # OR, if max distance moved > 15 pixels
        #print("average dist is: " + str(average_dist))
        if average_dist < average_thresh or max_dist < max_thresh:
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = obj3D['coords']
            else:
                all_voxels_kept = np.append(all_voxels_kept, obj3D['coords'], axis = 0)
            
            num_kept = num_kept + 1
        else:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            
        print("Finished distance thresholding for: " + str(num_elim + num_kept) + " out of " + str(len(cc_overlap_3D)) + " images")
    
    final_bleb_matrix = convert_vox_to_matrix(all_voxels_kept, np.zeros(all_blebs_THRESH.shape))
    elim_matrix = convert_vox_to_matrix(all_voxels_elim, np.zeros(all_blebs_THRESH.shape))
    print('Kept: ' + str(num_kept) + " eliminated: " + str(num_elim))
    
    return final_bleb_matrix, elim_matrix


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
s_path = './Checkpoints/2nd_OPTIC_NERVE_run_full_dataset/'
#s_path = './Checkpoints/3rd_OPTIC_NERVE_large_network/'

## for input
#input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/'
input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.16/EAE_A3/'
#input_path = './2018.08.16/EAE_A3/'
#input_path = './Training Data/'


""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
examples = [dict(input=i,truth=i.replace('.tif','truth.tif')) for i in images]

if truth:
    images = glob.glob(os.path.join(input_path,'*_pos_input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]

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
num_check = 365000
#num_check = 45000
checkpoint = '_' + str(num_check)
saver.restore(sess, s_path + 'check' + checkpoint)
    

# Required to initialize all
batch_size = 1;

batch_x = []; batch_y = [];
weights = [];

plot_jaccard = [];

output_stack = [];
output_stack_masked = [];
all_PPV = [];
input_im_stack = [];
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
                
                # some reasons values are switched in Barbara's images
                if "_BARBARA_" in truth_name:
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
        #input_cp = np.copy(input_save)
        #input_cp[seg_train == 0] = 0
        
        # convert to SINGLE channel + uint8, then apply threshold
        #input_cp = np.asarray(input_cp[:, :, 1], dtype=np.uint8)
        #th2 = cv.adaptiveThreshold(input_cp,255,cv.ADAPTIVE_THRESH_MEAN_C,\
        #    cv.THRESH_BINARY,11,2)
        #ret,thresh1 = cv.threshold(input_cp,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # use thresholded mask to set things to zero in the original seg output
        #seg_train_masked = np.copy(seg_train)
        #seg_train_masked[th2 == 0] = 0
        
        # clean by eroding then dilating
        #kernel = np.ones((5,5),np.uint8)
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
        #seg_train_dil = cv.morphologyEx(np.asarray(seg_train_masked, dtype=np.uint8), cv.MORPH_OPEN, kernel)
        #plt.imsave(sav_dir + filename_split + '_' + str(i) + '_output_mask_MASKED.tif', (seg_train_dil), cmap='binary_r')



        """ Compute accuracy """
        if truth:
            overlap_im = seg_train_dil + truth_im[:, :, 1]
            binary_overlap = overlap_im > 0
            labelled = measure.label(binary_overlap)
            cc_overlap = measure.regionprops(labelled, intensity_image=overlap_im)

            """ (1) Find # True Positives identified (overlapped) """
            masked = np.zeros(seg_train_dil.shape)
            all_no_overlap = np.zeros(seg_train_dil.shape)
            truth_no_overlap = np.zeros(seg_train_dil.shape)   # ALL False Negatives
            seg_no_overlap = np.zeros(seg_train_dil.shape)     # All False Positives
            for M in range(len(cc_overlap)):
                overlap_val = cc_overlap[M]['MaxIntensity']
                overlap_coords = cc_overlap[M]['coords']
                if overlap_val > 1:    # if there is overlap
                    for T in range(len(overlap_coords)):
                        masked[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]   # TRUE POSITIVES
                else:  # no overlap
                    for T in range(len(overlap_coords)):
                        all_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
                        truth_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     
                        seg_no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = overlap_im[overlap_coords[T,0], overlap_coords[T,1]]     

            labelled = measure.label(masked)
            cc_masked_TPs = measure.regionprops(labelled)
            TP = len(cc_masked_TPs)
                     
            labelled = measure.label(truth_no_overlap)
            cc_truth_FNs = measure.regionprops(labelled)
            FN = len(cc_truth_FNs)

            labelled = measure.label(seg_no_overlap)
            cc_seg_FPs = measure.regionprops(labelled)
            FP = len(cc_seg_FPs)
                        
            PPV = TP/(TP + FP)
                
            print("PPV value for image %d is: %.3f" %(i + 1, PPV))            
            all_PPV.append(PPV)
                        
#            """ (2) Find # of False Positives (Identified by """
#            """ part of the above cross_val"""
#            def compute_measures(TN, FP, FN, TP):
#                 """Computes effectiveness measures given a confusion matrix."""
#                 PPV = TP/(TP + FP)
#                 NPV = TN/(TN + FN)
#                 sensitivity = TP/(TP + FN)    # true positive rate
#                 specificity = TN/(TN + FP)    # true negative rate
#                 accuracy = (TP + TN) / (TP + FP + FN + TN)
#                 fmeasure = 2 * (specificity * sensitivity) / (specificity + sensitivity)
#            
#                 return PPV, NPV, sensitivity, specificity, accuracy, fmeasure
                    
                    



        """ Save as 3D stack """
        if len(output_stack) == 0:
            output_stack = seg_train
            #output_stack_masked = seg_train_masked
            input_im_stack = input_save[:, :, 1]
        else:
            output_stack = np.dstack([output_stack, seg_train])
            #output_stack_masked = np.dstack([output_stack_masked, seg_train_masked])
            input_im_stack = np.dstack([input_im_stack, input_save[:, :, 1]])



""" Pre-processing """
# 1) get more data (and good data)
# 2) overlay seg masks and binarize to get better segmentations???

    
""" Post-processing """
""" (1) removes all things that do not appear in > 5 slices!!!"""
all_seg, all_blebs, all_eliminated = slice_thresh(output_stack, slice_size=5)


filename_split = filename_split.split('_z')[0]

# Save output as individual .tiffs so can do stack later in imageJ
for i in range(len(output_stack[0, 0, :])):
    print("Printing post-processed output: " + str(i) + " of total: " + str(len(output_stack[0, 0, :])))
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_ORIGINAL_post-processed.tif', (all_seg[:, :, i]), cmap='binary_r')
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_BLEBS_post-processed.tif', (all_blebs[:, :, i]), cmap='binary_r')
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_ELIM_post-processed.tif', (all_eliminated[:, :, i]), cmap='binary_r')


""" (2) DO THRESHOLDING TO SHRINK SEGMENTATION SIZE, but do THRESH ON 3D array!!! """
save_input_im_stack = np.copy(input_im_stack)
input_im_stack[all_blebs == 0] = 0     # FIRST MASK THE ORIGINAL IMAGE

# then do thresholding
from skimage import filters
val = filters.threshold_otsu(input_im_stack)
mask = input_im_stack > val

# closes image to make less noisy """
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
blebs_opened_masked = cv.morphologyEx(np.asarray(mask, dtype=np.uint8), cv.MORPH_CLOSE, kernel)


""" apply slice THRESH again"""
all_seg_THRESH, all_blebs_THRESH, all_eliminated_THRESH = slice_thresh(blebs_opened_masked, slice_size=5)
# Save output as individual .tiffs so can do stack later in imageJ
for i in range(len(output_stack[0, 0, :])):
    print("Printing post-processed output: " + str(i) + " of total: " + str(len(output_stack[0, 0, :])))
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_THRESH_and_SLICED_post-processed.tif', (all_blebs_THRESH[:, :, i]), cmap='binary_r')
    

""" (3) Find vectors of movement and eliminate blobs that migrate """
final_bleb_matrix, elim_matrix = distance_thresh(all_blebs_THRESH, average_thresh=15, max_thresh=15)

# Save output as individual .tiffs so can do stack later in imageJ
for i in range(len(output_stack[0, 0, :])):
    print("Printing post-processed output: " + str(i) + " of total: " + str(len(output_stack[0, 0, :])))
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_DISTANCE_THRESHED_8px_post-processed.tif', (final_bleb_matrix[:, :, i]), cmap='binary_r')
    plt.imsave(sav_dir + filename_split + '_z' + str(i) + '_DISTANCE_THRESHED_8px_elimed_post-processed.tif', (elim_matrix[:, :, i]), cmap='binary_r')


""" Pseudo-local thresholding (applies Otsu to each individual bleb) """
#binary_overlap = all_blebs_THRESH > 0
#labelled = measure.label(binary_overlap)
#cc_overlap = measure.regionprops(labelled)
#
#total_blebs = 0
#pseudo_threshed_stack = np.zeros(np.shape(input_im_stack))
#for bleb in cc_overlap:
#    cur_bleb_coords = bleb['coords']
#    cur_bleb_mask = convert_vox_to_matrix(cur_bleb_coords, np.zeros(output_stack.shape))
#    
#    val = filters.threshold_otsu(cur_bleb_mask)
#    mask = cur_bleb_mask > val    
#    
#    pseudo_threshed_stack = pseudo_threshed_stack + mask
#
#    total_blebs = total_blebs + 1    
#        
#    print("Total analyzed: " + str(total_blebs) + "of total blebs: " + str(len(cc_overlap)))
    




""" Plotting as interactive scroller """
fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, elim_matrix)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()


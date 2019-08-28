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



# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(1); numpy.random.seed(1)

"""  Network Begins:
"""
## for saving
#s_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Checkpoints/3rd_run_SHOWCASE/'
#s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/3rd_OPTIC_NERVE_large_network/'
#s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/2nd_OPTIC_NERVE_run_full_dataset/'
#s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/4th_OPTIC_NERVE_4_deep_network/'
s_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Checkpoints/5th_OPTIC_NERVE_pos_only/'


## for input
#input_path = 'J:/DATA_2017-2018/Optic_nerve/EAE_miR_AAV2/2018.08.07/ON_11/Training Data/'
input_path = 'C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Training Data/'


""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*_pos_input.tif'))
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]

counter = list(range(len(examples)))  # create a counter, so can randomize it
counter = np.array(counter)
np.random.shuffle(counter)

val_size = 0.1;
val_idx_sub = round(len(counter) * val_size)
validation_counter = counter[-1 - val_idx_sub : -1]
input_counter = counter[0: -1 - val_idx_sub]

# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 1024, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name = 'weighted_labels')


""" Creates network and cost function"""
#y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network_SMALL(x, y_, training)
#y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network_4_layers(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = [ f for f in listdir(s_path) if isfile(join(s_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_check.sort(key = natsort_key1)

""" If no old checkpoint then starts fresh """
if len(onlyfiles_check) < 8:  
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 
    plot_cost = []; plot_cost_val = []; plot_jaccard = []; plot_jaccard_val = [];
    num_check= 0;

else:   
    """ Find last checkpoint """   
    last_file = onlyfiles_check[-11]
    split = last_file.split('.')
    checkpoint = split[0]
    num_check = checkpoint.split('_')
    num_check = int(num_check[1])
    
    saver.restore(sess, s_path + checkpoint)
    
    # Getting back the objects:
    with open(s_path + 'loss_global.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_cost = loaded[0]
    
    # Getting back the objects:
    with open(s_path + 'loss_global_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_cost_val = loaded[0]  
    
    # Getting back the objects:
    with open(s_path + 'jaccard.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_jaccard = loaded[0] 
    
    # Getting back the objects:
    with open(s_path + 'jaccard_val.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_jaccard_val = loaded[0] 
    
    # Getting back the objects:
    with open(s_path + 'val_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        validation_counter = loaded[0]     
        
    # Getting back the objects:
    with open(s_path + 'input_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        input_counter = loaded[0]  

# Required to initialize all


batch_size = 4; 
save_epoch = 1000;
plot_every = 10;
epochs = num_check;

batch_x = []; batch_y = [];
weights = [];

for P in range(8000000000000000000000):

    np.random.shuffle(input_counter)
    for i in range(len(input_counter)):
        
        input_name = examples[input_counter[i]]['input']
        input_im = np.asarray(Image.open(input_name), dtype=np.float32)
        
        """ maybe remove normalization??? """
        input_im = normalize_im(input_im, mean_arr, std_arr) 
        
        truth_name = examples[input_counter[i]]['truth']
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
#            truth_im[:, :, 0] = channel_2   # background
#            truth_im[:, :, 1] = channel_1   # blebs
#            
#            # some reasons values are switched in Barbara's images
#            if "_BARBARA_" in truth_name:
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
        
        """ set inputs and truth """
        batch_x.append(input_im)
        batch_y.append(truth_im)
        weights.append(weighted_labels)
                
        
        """ Feed into training loop """
        if len(batch_x) == batch_size:
           feed_dict_TRAIN = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 

           train_step.run(feed_dict=feed_dict_TRAIN)

           batch_x = []; batch_y = []; weights = [];
           epochs = epochs + 1           
           print('Trained: %d' %(epochs))
           
           
           if epochs % plot_every == 0:
              plt.close(1)
              plt.close(2)
              plt.close(18)
              plt.close(19)
              plt.close(21)
              batch_x_val = []
              batch_y_val = []
              batch_weights_val = []
              np.random.shuffle(validation_counter)
              for batch_i in range(len(validation_counter)):
                  """ GET VALIDATION, almost exact same as above except use validation_counter as counter"""
                  input_name = examples[validation_counter[batch_i]]['input']
                  input_im_val = np.asarray(Image.open(input_name), dtype=np.float32)
                
                  truth_name = examples[validation_counter[batch_i]]['truth']
                  truth_tmp_val = np.asarray(Image.open(truth_name), dtype=np.float32)
                              
                  """ maybe remove normalization??? """
                  input_im_val = normalize_im(input_im_val, mean_arr, std_arr) 
               
                  """ convert truth to 2 channel image """
                  if "_neg_" in truth_name:
                      truth_im_val = np.zeros(np.shape(truth_tmp_val) + (2,))
                      truth_im_val[:, :, 0] = np.ones(np.shape(truth_tmp_val))   # background
                      truth_im_val[:, :, 1] = np.zeros(np.shape(truth_tmp_val))   # blebs
                  else:
                      channel_1 = np.copy(truth_tmp_val)
                      channel_1[channel_1 == 0] = 1
                      channel_1[channel_1 == 255] = 0
                                
                      channel_2 = np.copy(truth_tmp_val)
                      channel_2[channel_2 == 255] = 1   
                          
                      truth_im_val = np.zeros(np.shape(truth_tmp_val) + (2,))
#                      truth_im_val[:, :, 0] = channel_2   # background
#                      truth_im_val[:, :, 1] = channel_1   # blebs
#                      
#                      # some reasons values are switched in Barbara's images
#                      if "_BARBARA_" in truth_name:
                      truth_im_val[:, :, 0] = channel_1   # background
                      truth_im_val[:, :, 1] = channel_2   # blebs

                     
        
                  blebs_label = np.copy(truth_im_val[:, :, 1])
                 
                  """ Get spatial AND class weighting mask for truth_im """
                  sp_weighted_labels_val = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
                
                  """ OR DO class weighting ONLY """
                  #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
                
                  """ Create a matrix of weighted labels """
                  weighted_labels_val = np.copy(truth_im_val)
                  weighted_labels_val[:, :, 1] = sp_weighted_labels_val
                
                  """ set inputs and truth """
                  batch_x_val.append(input_im_val)
                  batch_y_val.append(truth_im_val)
                  batch_weights_val.append(weighted_labels_val)
                  
                  if len(batch_x_val) == batch_size:
                      break

             
              feed_dict_CROSSVAL = {x:batch_x_val, y_:batch_y_val, training:0, weight_matrix:batch_weights_val}      
              
              """ Training loss"""
              loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
              plot_cost.append(loss_t);                 
                
              """ Training Jaccard """
              jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN)
              plot_jaccard.append(jacc_t)           
              
              """ CV loss """
              loss_val = cross_entropy.eval(feed_dict=feed_dict_CROSSVAL)
              plot_cost_val.append(loss_val)
             
              """ CV Jaccard """
              jacc_val = jaccard.eval(feed_dict=feed_dict_CROSSVAL)
              plot_jaccard_val.append(jacc_val)
              
              """ function call to plot """
              plot_cost_fun(plot_cost, plot_cost_val)
              plot_jaccard_fun(plot_jaccard, plot_jaccard_val)
              
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(21); plt.savefig(s_path + 'jaccard.png')

              """ Plot for debug """
              batch_x.append(input_im); batch_y.append(truth_im); weights.append(weighted_labels);
              feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
              output_train = softMaxed.eval(feed_dict=feed_dict)
              seg_train = np.argmax(output_train, axis = -1)[0]              
              
              batch_x = []; batch_y = []; weights = [];
              batch_x.append(input_im_val); batch_y.append(truth_im_val); weights.append(weighted_labels_val);
              feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
              output_val = softMaxed.eval(feed_dict=feed_dict)
              seg_val = np.argmax(output_val, axis = -1)[0]    
              
              
              plt.figure(num=2, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
              plt.subplot(331); plt.imshow(truth_im[:, :, 1]); plt.title('Truth Train');
              plt.subplot(332); plt.imshow(seg_train); plt.title('Output Train');              
              plt.subplot(334); plt.imshow(truth_im_val[:, :, 1]); plt.title('Truth Validation');        
              plt.subplot(335); plt.imshow(seg_val); plt.title('Output Validation'); plt.pause(0.0005);

              #plt.subplot(333); plt.imshow(np.asarray(input_im, dtype = np.uint8)); plt.title('Input');
              plt.subplot(333); plt.imshow(sp_weighted_labels); plt.title('weighted');    plt.pause(0.005)
              plt.subplot(336); plt.imshow(truth_im[:, :, 0]); plt.title('Ch1: background');
              plt.subplot(339); plt.imshow(truth_im[:, :, 1]); plt.title('Ch2: blebs');       
              plt.pause(0.05)

              
              if epochs > 500:
                  if epochs % 500 == 0:
                      plt.savefig(s_path + '_' + str(epochs) + '_output.png')
              elif epochs % 10 == 0:
                  plt.savefig(s_path + '_' + str(epochs) + '_output.png')
              
              batch_x = []; batch_y = []; weights = [];
              
           """ To save (every x epochs) """
           if epochs % save_epoch == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_' +  str(epochs)
              save_path = saver.save(sess_get, save_name)
               
              """ Saving the objects """
              save_pkl(plot_cost, s_path, 'loss_global.pkl')
              save_pkl(plot_cost_val, s_path, 'loss_global_val.pkl')
              save_pkl(plot_jaccard, s_path, 'jaccard.pkl')
              save_pkl(plot_jaccard_val, s_path, 'jaccard_val.pkl')
              save_pkl(validation_counter, s_path, 'val_counter.pkl')
              save_pkl(input_counter, s_path, 'input_counter.pkl')   
                                            
              """Getting back the objects"""
              plot_cost = load_pkl(s_path, 'loss_global.pkl')
              plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
              plot_jaccard = load_pkl(s_path, 'jaccard.pkl')
              plot_jaccard_val = load_pkl(s_path, 'jaccard_val.pkl')
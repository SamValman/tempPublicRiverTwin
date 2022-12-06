# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:21:14 2022

@author: lgxsv2

"""


#%%TRAINING THE CNN
import os
# below is different from runable file (Temporary sharable repo)
os.chdir(r'D:\Code\RiverTwin\2022_10_11_packageComparison')
# load functions 
from TrainRiverTwinWaterMask import Train_RiverTwinWaterMask, GPU_SETUP
#check gpu
GPU_SETUP()

#%%

# FolderContents only needed if saving a new training dataset
# New datasets are saved as collection of tiled tifs
FolderContents = ['10_A1.tif', '10_A2.tif', '10_T1.tif', '10_T2.tif', '10_V1.tif', '10_V2.tif', '11_B1.tif', '11_B2.tif', '11_Po1.tif', '11_Po2.tif', '11_Py1.tif', '11_Py2.tif', '12_C1.tif', '12_C2.tif', '12_E1.tif', '12_E2.tif', '12_T1.tif', '12_T2.tif', '13_A1.tif', '13_A2.tif', '13_R1.tif', '13_R2.tif', '13_S1.tif', '13_S2.tif', '1_A1.tif', '1_A2.tif', '1_B1.tif', '1_B2.tif', '1_X1.tif', '1_X2.tif', '2_B1.tif', '2_B2.tif', '2_N1.tif', '2_N2.tif', '2_R1.tif', '2_R2.tif', '4_R1.tif', '4_R2.tif', '4_Th1.tif', '4_Th2.tif', '4_Tr1.tif', '4_Tr2.tif', '5_C1.tif', '5_C2.tif', '5_H1.tif', '5_H2.tif', '5_S1.tif', '5_S2.tif', '6_M1.tif', '6_M2.tif', '6_S1.tif', '6_S2.tif', '6_T1.tif', '6_T2.tif', '7_C1.tif', '7_C2.tif', '7_N1.tif', '7_N2.tif', '7_V1.tif', '7_V2.tif', '8_M1.tif', '8_M2.tif', '8_R1.tif', '8_R2.tif', '8_V1.tif', '8_V2.tif', '9_B1.tif', '9_B2.tif', '9_N1.tif', '9_N2.tif', '9_T1.tif', '9_T2.tif']

#trainingfolder to load premade dataset from (numeric is the tile size) 
trainingfolder = 'D:/Training_data/temporary_tiles/Balanced32'


#%%
# original attempt - doesn't appear to use GPU despite printing it out. 
# memory leakage is the main issue
# can ignore first 4 arguents as not creating new training data
Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents,
                         balanceTrainingData=1, trainingFolder=trainingfolder,
                         outfile='2022_11_29_foursaved\Balanced32',
                          epochs=20, bs=256, lr_type='plain',
                          tileSize=32)


#%%
#below utilises depreciated tf comands 
# appears to work on GPU same memory issues
import tensorflow as tf

with tf.compat.v1.Session as sess:
    Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents,
                         balanceTrainingData=1, trainingFolder=trainingfolder,
                         outfile='2022_11_29_foursaved\Balanced32',
                          epochs=20, bs=256, lr_type='plain',
                          tileSize=32)









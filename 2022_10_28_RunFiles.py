# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:21:14 2022

@author: lgxsv2


###THE FIRST HALF OF THIS SCRIPT WIL BE MOVED TO AN ARCHIVE TESTING FOLDER, 
THE CNN SHOULD NOT NEED TO BE RE RAN AFTER ITS FINAL CHOICE
"""


import os
import glob
import pandas as pd


os.chdir(r'D:\Code\RiverTwin\2022_10_11_packageComparison')
from TrainRiverTwinWaterMask import  GPU_SETUP
from RiverTwinWaterMask import RiverTwinWaterMask
from testSuccess import testSuccess

GPU_SETUP()


#%%

imPath = r"D:\Training_data\test\*.tif" 
fn_model = r"D:\Code\RiverTwin\ZZ_ModelOutputs\try20\model"
output = r'D:\Code\RiverTwin\ZZ_Results\2022_11_17_try20'

names = []
f1s = []

# iterate through images
for i in glob.glob(imPath):

    
    im_name = i.split('\\')[-1]
    print(im_name)
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=20,
                                            output=output)
    # p3 = os.path.join(output, 'p3', i.split('\\')[-1])
    r = testSuccess(p3, output, time)
    
    names.append(im_name[:-4])
    f1s.append(r[3][1])
    
#%% Save f1s 
# df = pd.DataFrame({'id':names,'f1':f1s })

# fn = 'D:\\Code\\RiverTwin\\ZZ_ModelOutputs\\VGG16\\notF1s.csv'
# df.to_csv(fn)
    



raise SystemExit()


#%%
imPath = r"D:\Code\RiverTwin\ZZ_Results\2022_11_11_VGG16\p3\*.tif"
output = r'D:\Code\RiverTwin\ZZ_Results\2022_11_11_VGG16'

for i in glob.glob(imPath):
    
    testSuccess(i, output=output, time=False, display_image=True, save_image=True)




#%% Run OTSU
from otsuPackage import otsuPackage
import glob

imPath = r"D:\Training_data\test"
output = r'D:\Code\RiverTwin\ZZ_Results\2022_11_15_OTSU'

# otsuPackage(folder=imPath, op_folder=output)

imPath = r'D:\Code\RiverTwin\ZZ_Results\2022_11_15_OTSU\o1\*.tif'
output = r'D:\Code\RiverTwin\ZZ_Results\2022_11_15_OTSU'


for i in glob.glob(imPath):
    testSuccess(i, output=output, time=False, display_image=True, save_image=True)
    




#%%
import skimage.io as IO
import matplotlib.pyplot as plt
import numpy as np 

for i in glob.glob(output):
    im = IO.imread(i)
    break

im = i1-im



#%%
from ANNPackage import ANNPackage, TrainANNPackage

imFP = r'D:\Training_data\train\*.tif'
lbFP = r'D:\Training_data\label_train\*.tif'

TrainANNPackage(imFP, lbFP)









#%%


fp = ''
import os
os.chdir(r'D:\Code\RiverTwin\2022_10_11_packageComparison')
from testSuccess import testSuccess


testSuccess(fp, r'D:\Code\RiverTwin\ZZ_Results2_11_03_Base\p3')



























###%%
#%%TRAINING THE CNN
import os
os.chdir(r'D:\Code\RiverTwin\2022_10_11_packageComparison')
from TrainRiverTwinWaterMask import Train_RiverTwinWaterMask, GPU_SETUP
GPU_SETUP()

#%%


FolderContents = ['10_A1.tif', '10_A2.tif', '10_T1.tif', '10_T2.tif', '10_V1.tif', '10_V2.tif', '11_B1.tif', '11_B2.tif', '11_Po1.tif', '11_Po2.tif', '11_Py1.tif', '11_Py2.tif', '12_C1.tif', '12_C2.tif', '12_E1.tif', '12_E2.tif', '12_T1.tif', '12_T2.tif', '13_A1.tif', '13_A2.tif', '13_R1.tif', '13_R2.tif', '13_S1.tif', '13_S2.tif', '1_A1.tif', '1_A2.tif', '1_B1.tif', '1_B2.tif', '1_X1.tif', '1_X2.tif', '2_B1.tif', '2_B2.tif', '2_N1.tif', '2_N2.tif', '2_R1.tif', '2_R2.tif', '4_R1.tif', '4_R2.tif', '4_Th1.tif', '4_Th2.tif', '4_Tr1.tif', '4_Tr2.tif', '5_C1.tif', '5_C2.tif', '5_H1.tif', '5_H2.tif', '5_S1.tif', '5_S2.tif', '6_M1.tif', '6_M2.tif', '6_S1.tif', '6_S2.tif', '6_T1.tif', '6_T2.tif', '7_C1.tif', '7_C2.tif', '7_N1.tif', '7_N2.tif', '7_V1.tif', '7_V2.tif', '8_M1.tif', '8_M2.tif', '8_R1.tif', '8_R2.tif', '8_V1.tif', '8_V2.tif', '9_B1.tif', '9_B2.tif', '9_N1.tif', '9_N2.tif', '9_T1.tif', '9_T2.tif']
trainingfolder = 'D:/Training_data/temporary_tiles/Balanced32'


#%%
Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents,
                         balanceTrainingData=1, trainingFolder=trainingfolder,
                         outfile='2022_11_29_foursaved\Balanced32',
                          epochs=20, bs=32, lr_type='plain',
                          tileSize=32)























#%%
# Train_RiverTwinWaterMask(newTrainingData=True, trainingData=FolderContents, trainingFolder=trainingfolder, outfile='2022_11_04_TS20',
#                           epochs=40, bs=64, 
#                           tileSize=20)
Train_RiverTwinWaterMask(newTrainingData=True, trainingData=FolderContents, balanceTrainingData=1.25, trainingFolder=trainingfolder, outfile='randExtra',
                          epochs=40, bs=64, lr_type='plain',
                          tileSize=32)
Train_RiverTwinWaterMask(newTrainingData=True, trainingData=FolderContents, balanceTrainingData=1, trainingFolder=trainingfolder, outfile='realBalanceExtra',
                          epochs=40, bs=64, lr_type='plain',
                          tileSize=32)
# Train_RiverTwinWaterMask(newTrainingData=True, trainingData=FolderContents, balanceTrainingData=1.25, trainingFolder=trainingfolder, outfile='MultiTile_ts80',
#                           epochs=40, bs=64, lr_type='other',
#                           tileSize=80)

# Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents, balanceTrainingData=1.25, trainingFolder=trainingfolder, outfile='2022_11_08_BalancedNormal',
#                           epochs=100, bs=64, 
#                           tileSize=32)

Train_RiverTwinWaterMask(newTrainingData=True, trainingData=FolderContents,
                         trainingFolder=trainingfolder,
                         balanceTrainingData=2, 
                         loss_type='focal', alpha=0.5, gamma=2, 
                         outfile='2022_11_08_focal_A0_5_B2_noprune',
                          epochs=40, bs=64, 
                          )


#%%
#worse
# Train_RiverTwinWaterMask(newTrainingData=False, inc_neck=True, trainingFolder=trainingfolder, outfile='2022_11_01_base_neck')
# #too busy
# Train_RiverTwinWaterMask(newTrainingData=False, inc_2ndbatch=True, inc_neck=True, trainingFolder=trainingfolder, outfile='2022_11_01_base_neckandback')


















#%% Test problem

# workflowPlot(band_im, CNNPrediction_argmaxed, ANNPrediction_argmaxed, finalResult, op='.png', save='no')
#%%


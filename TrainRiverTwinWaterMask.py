# -*- coding: utf-8 -*-
"""
###############################################################################
Project: River Twin: WaterMask
###############################################################################
Final River twin water mask model
Created on Wed Oct 26 16:39:34 2022

@author: lgxsv2
"""
#%% packages

import numpy as np
import random
import math
import pandas as pd

#plotting
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


# should probably only use one keras layers style
from tensorflow import keras
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import callbacks
from keras.applications.vgg16 import VGG16

import skimage.io as IO
import glob
import os
import datetime
import shutil


import tensorflow_addons as tfa
import gc
#%% Wrapper


def Train_RiverTwinWaterMask(newTrainingData=True, trainingFolder='',
                             trainingData=[], balanceTrainingData=1, tileSize=32,
                             epochs=100, bs=6,
                             lr=0.001, lr_type='plain', 
                             loss_type='notfocal', alpha=1, gamma=2,
                             inc_2ndbatch=False, inc_neck=False,
                             outfile='todaysmodel', 
                             saveModel=True
                             ):
    '''
    Trains a CNN to provide the basis for the cCNN water mask.
    Code format is Section number, letter, number. in a nested format. 
    
    Patches needed:
        no extra file path for multiple different balanced runs in a day.
        
        
    Parameters
    ----------
    trainingData : TYPE
        DESCRIPTION.
    balanceTrainingData: int
        Default: None
        if a value it will balance training data.
        1 == balance, >1 == more land than water, <1 == more water than land.
    hyperParameters : dict
        {epochs:int, 'batchsize':int, }.
    tileSize : Int
        tile size used to create pure training tiles
    
    ## Needs to know if to balance or not ## 

    Returns
    -------
    None.

    '''
    ### MODEL PARAMETERS
    # uncomment if fixed
    # tileSize = 32
    # balanceTrainingData = 1
    
    
    #Section One:  Training data 
    if newTrainingData: 
        Xy_train, Xy_test = CollectAndOrganiseCNNTrainingData(trainingData, 
                                      tileSize, balanceTrainingData, bs, epochs)
        # the above needs to return a file path and close X_train, y_train
    else:
        print('Using pre-collected training data')
        Xy_train, Xy_test = reloadTrainingData(trainingFolder, epochs, bs)
    
    #Section Two: Train CNN
    # change as needed 
    CNN = fit_CNN(Xy_train, Xy_test, epochs, lr, lr_type, loss_type, alpha, gamma, inc_2ndbatch, inc_neck) #epochs, bs,

    #Section Three: Save model
    saveModelAndOutputs(CNN, saveModel, outfile)
    
    
    
###############################################################################   
#%% Section one: Wrapper. 
###############################################################################

def CollectAndOrganiseCNNTrainingData(trainingData, tileSize,
                                      balanceTrainingData, bs, epoch):
    '''
    Wrapper to do all the work to collect and organise training data
    
    Parameters
    ----------
    trainingData : list
        list of file names of rivers in the training folder specified .
    tile_size : int
        specified in Train_RiverTwinWaterMask wrapper.
    balanceTrainingData: int
        Degree of training data inbalancing acceptable. 

    Returns
    -------
    X_train : NumpyArray
        data ready for training.
    y_train : NumpyArray
        labels ready for training.

    '''
    ## Section one:A - removes training files wihtout labels
    im_list, label_list = ListUsableFileNames(trainingData)
    print('Total number of images: ', len(im_list))
    
    ## Section one:B - creates pure tiles to train model
    # 0==water, 1==land.
    X_train, y_train = stackTiles(im_list, label_list, tileSize)
    
    # balance training data as requested (saves regardless)
    ## Section one: c function to save and balance training data
    Xy_train, Xy_test = pruneAndSave(X_train, y_train,
                                        balanceTrainingData, tileSize, epoch, bs)
        
        
        
        
    
    return Xy_train, Xy_test


#%% Section one: A: 
    
def ListUsableFileNames(trainingData):
    '''
    Checks if images have labels and returns the file path of those that do
    
    Parameters
    ----------
    trainingData : list
        Training images we would like to use.
    

    Returns
    -------
    im_list : list
        list of filepaths for available X_train images.
    label_list : list
        list of filepaths for available y_train label images.
.

    '''
    im_list, label_list = [], []
    
    # list all label files available and cut to just their names
    file_ls =  glob.glob(os.path.join('D:/Training_data/label_train/', '*.tif'))
    file_ls = [x[29:] for x in file_ls]
    
    for i in trainingData:
        # checks if that image has a label
        if i in file_ls:
            # A :1 function
            im, label = getPaths(i)

            # join to list for return
            im_list.append(im)
            label_list.append(label)

        
    return im_list, label_list   

### A: 1
def getPaths(riverID):
    '''
    gets paths for river images and labels 
    Parameters
    ----------
    riverName : str
        river Image ID name.

    Returns
    -------
    im_ls : list
        list of image filepaths .
    label_ls : list
        list of label image filepaths.

    '''
    # riverID = riverID +'.tif'
    imPath = os.path.join(r'D:/Training_data/train/', riverID) 
    labPath = os.path.join(r'D:/Training_data/label_train/', riverID) 
    

    return imPath, labPath


###############################################################################    
#%% Section one: B
###############################################################################

def stackTiles(im_list, label_list, tile_size):
    '''
   Creates (N, tileSize, tileSize, bandNumber) arrays
   from training and label data (0=water, 1 = land)
   
    Parameters
    ----------
    im_list : List
        Training image file paths.
    label_list : List
        Label image file paths.
    tile_size : Int
        Tile size to be maintained throughout.

    Returns
    -------
    X_train : numpy array
    
    y_train : numpy array

    '''
    # empty lists to be filled 
    X_train, y_train = [],[]
    
    for i,m in zip(im_list, label_list):
        #open label image
        label_im = np.int16(IO.imread(m))
        # check that the image is not too large
        # skips if this is the case
        # Gdal options available but not built in 
        try:
            #open related training image
            band_im = np.int16(IO.imread(i))
        except MemoryError:
            print(i, 'too large')
            continue
       
        
        #Section 1:B:1 function to turn individual images to tiles
        X_temp, y_temp = tileForCNN(band_im, label_im, tile_size)
        
        
        # print the image name to show it worked
        print(i)
      
        # remove empyt images - str output from tileForCNN
        if type(X_temp) != str:
            X_train.append(X_temp)
            y_train.append(y_temp)
        else:
            print(X_temp)
            continue
    
    # combine lissts into format (N, tileSize, tileSize, bandNumber)
    # band number is binary for y_train
    X_train = np.concatenate((X_train), axis=0)
    y_train = np.concatenate((y_train), axis=0)
    
    return X_train, y_train

### B:1
def tileForCNN(im, label, tileSize):
    '''
    cuts individual images into tiles 
    Parameters
    ----------
    im : array
        Band image (4 band to be used).
    label : array
        label image (0, 1, 2 for nothing, water, land respectively).
    tileSize: int
        from uppermost function
   
    Returns
    -------
    tiled image as input to CNN.
    '''
    # normalise satellite bands
    # normalised removed for storage etc - will put in with data reading
    # im = keras.utils.normalize(im)

    # remove edge cells from image so it can be tiled precisely
    height, length = label.shape
    height, length = int(height//tileSize), int(length//tileSize)
    y_axis, x_axis = (height*tileSize), (length*tileSize)
    
    # cut to just divisable area 
    # makes them divisable by tile size used later
    im = im[:y_axis, :x_axis,:]
    label = label[:y_axis, :x_axis]
    
    #int of empty tile size
    pure_tile = tileSize**2
   
    #list for output tiles
    im_ls = []
    label_ls = []
    
    # scrolls through image height and length (when multiplied by tile_size)
    # height len already//tilesize
    for m in range(height):
        for n in range(length):
            temp_m = (m+1)*tileSize
            temp_n = (n+1)*tileSize
            # selects this tile out of label image
            label_tile = label[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n]
            
            # only pure tiles  (water first)
            if np.count_nonzero(label_tile == 1) == pure_tile:
                # create tile with these values
                band_tile = im[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n, :]
                # captures tile errors if they occur 
                if band_tile.shape != (tileSize,tileSize,4):
                    continue
                # label is just a list so only needs int appended 
                im_ls.append(band_tile)
                label_ls.append(1)
    
            # land now - if pure all same as above. 
            elif np.count_nonzero(label_tile == 2) == pure_tile:
                band_tile = im[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n, :]
                if band_tile.shape != (tileSize,tileSize,4):
                    continue
                im_ls.append(band_tile)
                label_ls.append(2)
    
    # some images may have no tiles if they were very small or very unpure.
    if len(im_ls)!= 0:        
        im_ls = np.array(im_ls)
        label_ls = np.array(label_ls).reshape((-1,1))
    
        # get format correct 
        # now water 0, land 1 
        label_ls = label_ls - 1
    else: 
        im_ls, label_ls = 'No tiles', 'no Tiles'
                
    return im_ls, label_ls


#%% Section one:C
def pruneAndSave(X_train, y_train, balanceTrainingData, tileSize,epoch, bs):
    '''
    organises the balancing and saving of training data for the model

    Parameters
    ----------
    X_train : array
        all X_train.
    y_train : array
        all y_train.
    balanceTrainingData : int
        balance value.
    extra_folder_name : 'str', optional
        DESCRIPTION. The default is ''.
        for having more than one different training set in a day
    epoch: int
    bs : int
        
        

    Returns
    -------
    Xy_train : tf data dataset
        new X_train
    Xy_test: tf data dataset
        new y_train

    '''
    #create_training_directory_to_save_into
    parent_dir='D:/Training_data/temporary_tiles'
    
    #extra folder name for two in a day
    directory =  datetime.datetime.today().strftime('%Y_%m_%d')
    # +extra_folder_name If wanted it needs to be added in all levels
    path = os.path.join(parent_dir, directory)
    
   #remove dir if already exisits
    if os.path.exists(path):
        print('overwriting old directory from today')
        shutil.rmtree(path)
    os.mkdir(path)
    
    # add water and land label categories 
    water = os.path.join(path, 'water')
    land = os.path.join(path, 'land')
    
    os.mkdir(water)
    os.mkdir(land)
    name = 0
    
    # go through each tile and place in correct folder
    for tile, label in zip(X_train, y_train): 
        name +=1
        # 1:C:1 function for saving all tiles
        saveTile(tile, label, name, water, land)
    
    # No longer need so remove
    X_train, y_train = None, None
    #function 1:C:2
    # remove tiles from the larger class based on balanceTrainingData input
    prune(water, land, balanceTrainingData)
    
    # function 1:C:3
    Xy_train, Xy_test = reloadTrainingData(path, epoch, bs)
    
    return Xy_train, Xy_test
    
    
    
### section 1:C:1
def saveTile(tile, label, name, water_path, land_path):
    '''
    Saves the tiles (all)

    Parameters
    ----------
    tile : array
        individual tile.
    label : array
        tile label.
    name : int
        descriptor for that tile number.
    water_path : str
        folder path.
    land_path : str
        folder path.

    Raises
    ------
    SystemExit
        if there is an error it will shut the model down.

    Returns
    -------
    None.

    '''
    name = str(name) + '.tif'
    print('norm' , type(tile[1][1][1]))

    if label[0] == 0:
        temp_im_path = os.path.join(water_path, name)
                
        IO.imsave(temp_im_path, tile, check_contrast=False)
        
        #add spin and save function to increase training data 1:c:6
        spinAndSave(water_path, name, tile)
        
    elif label[0] == 1:
        temp_im_path = os.path.join(land_path, name)
        IO.imsave(temp_im_path, tile, check_contrast=False)
        
        #add spin and save function to increase training data 1:c:6
        spinAndSave(land_path, name, tile)

    else:
        print('error probably should start looking here')
        print('name')
        raise SystemExit()
       
        
       
### Section 1:C:2
def prune(water_folder_name, land_folder_name, balanceTrainingData):
    '''
    Deletes according to balanceTrainingData to balance training set

    Parameters
    ----------
    water_folder_name : str
        folder path.
    land_folder_name : str
        folder path.
    balanceTrainingData : int
        acceptable balance see wrapper function.

    Returns
    -------
    None.

    '''
    # how many water tiles are there
    water_tiles = glob.glob(water_folder_name + '/*.tif')
    water_tiles = len(water_tiles)
    
    # get all land tiles
    land_tiles = glob.glob(land_folder_name + '/*.tif')
    
    # using balanceTrainingData as a weight to increase or decrease tile overlap. 
    # balanceTrainingData of 1.1 would allow 10% more land than water
    water_tiles = int(water_tiles*balanceTrainingData)
    if water_tiles < len(land_tiles):
        # selects random land images up to number of water tiles. (pre-influenced)
        removable_images = np.random.choice(np.arange(len(land_tiles)), len(land_tiles)-water_tiles, replace=False)
        print('removing ',len(removable_images), ' land tiles to balance dataset')
       
        # actually do the removing
        for i in removable_images:
            os.remove(land_tiles[i])

### section 1:C:3
def reloadTrainingData(folder, epoch, bs):
    '''
    reloads dataset (trimmed if so) using tf Datasets
    This enables reading from disk and much higher batch sizes

    Parameters
    ----------
    folder: str
        name of data folder (level above water, land)
    epoch: int
        tf Datasets needs the epoch level here
    bs:int
        tf Datasets needs the epoch at this level 

    Returns
    -------
    Xy_train : tf.data.Dataset
        training dataset
    Xy_test : tf.data.Dataset
        testing dataset

    '''
    #get a file path that will find all the water and all the land images
    data_dir = folder + '/*/*'

    # create dataset using find file, shuffle
    ds = tf.data.Dataset.list_files(data_dir, shuffle=True)
    
    # Read the images and labels, add batchsize and number of epochs
    # requires 1:C:4, 1:C:5 to process_image and label
    ds = ds.shuffle(len(ds)).map(process_image).batch(bs)
    ds = ds.prefetch(1) #.repeat(epoch)

    # set validation size
    training_size = int(len(ds)*0.66)
    
    # splits datset into test and validation
    Xy_train = ds.take(training_size)
    Xy_test = ds.skip(training_size)
        



    return Xy_train, Xy_test
        
### Section 1:C:4
def process_image(file_path):
    '''

    Parameters
    ----------
    file_path : str
        mapped file path in DS.

    Returns
    -------
    im : array 
        in shape batchsize, batchsize, 4.
    label : array
        1 = land, 0 = water.

    '''
    # uses function s1:C:5
    label = get_label(file_path)
    
    # uses tf.io to decode tiff because not an option in base tf
    im = tf.io.read_file(file_path)
    im = tfio.experimental.image.decode_tiff(im)

    return im, label

### Section 1: C: 5
def get_label(file_path):
    '''
    gets label from folder and returns binary option

    Parameters
    ----------
    file_path : str
        file path from ds
        
    Returns
    -------
    label : array
        binary label from folder, 0 water, 1 land.

    '''
    # takes folder name (water, land)
    str_label = tf.strings.split(file_path, os.sep)[-2]
    
    #turns into tf friendly binary
    label = tf.cond(tf.math.equal(str_label, tf.constant('land', dtype=tf.string)), 
                    lambda:tf.constant(1, shape=(1,)), 
                    lambda:tf.constant(0, shape=(1,)))
    
    return label

### 1:c:6
def spinAndSave(label_path, name, tile):
    
    # letter prevents overwritting
    #Rotates tile 90 degrees
    tile=np.rot90(tile)
    #add small noise int to each
    t = tile + np.random.randint(low=-10, high=+10, size=tile.shape)
    name = 'a'+name
    temp_im_path = os.path.join(label_path, name)

    IO.imsave(temp_im_path, np.int16(t), check_contrast=False)
    
    #Rotates tile 180 degrees
    tile=np.rot90(tile)
    t = tile + np.random.randint(low=-10, high=+10, size=tile.shape)
    name = 'b'+name
    temp_im_path = os.path.join(label_path, name)
    IO.imsave(temp_im_path, np.int16(t), check_contrast=False)
    
    #Rotates tile 270 degrees
    tile=np.rot90(tile)
    t = tile + np.random.randint(low=-10, high=+10, size=tile.shape)

    name = 'c'+name
    temp_im_path = os.path.join(label_path, name)
    IO.imsave(temp_im_path,np.int16(t), check_contrast=False)
    


    # I=Tile+np.random.randint(low=-10, high=+10, size=Tile.shape) #Rotates tile 90 degrees + noise from -10 to 10.

###############################################################################
#%% Section 2: Train CNN    
###############################################################################

# BUILD all as complicated options then remove as needed
def fit_CNN(Xy_train, Xy_test, epochs,  lr,
               lr_type, loss_type, alpha, gamma, 
               inc_2ndbatch, inc_neck):
    '''
    fits a basic CNN
    Parameters
    ----------
    X_train : array
        shape [ntiles, ].
    y_train : array
        shape [ntiles, ].

    Returns
    -------
    CNN : Model
        trained model.

    '''
    # get shape from 1st tile 
    # for i,l in Xy_train.take(1):
    #     shape = i.numpy()[0].shape
    # TransferedModel = VGG16(include_top=False, weights=None, input_shape=(shape)) #input_shape=(40,40,4)
    # flat = keras.layers.Flatten()(TransferedModel.layers[-1].output)
    # class1 = keras.layers.Dense(32, activation='relu')(flat) 
    # output = keras.layers.Dense(2, activation='softmax')(class1)
    
    
    # CNN = keras.Model(inputs=TransferedModel.inputs, outputs=output)
     
    
    # CNN.compile(loss="sparse_categorical_crossentropy", 
    #               optimizer="adam",
    #               metrics=["accuracy"])

    # Sequential model
    CNN = keras.Sequential()
    
    # input shape #
    # get shape from 1st tile 
    for i,l in Xy_train.take(1):
        shape = i.numpy()[0].shape
    CNN.add(keras.Input(shape=shape))
   
    #Need to normalize here as layer
    CNN.add(tf.keras.layers.Normalization())

    

    CNN.add(Conv2D(64, (3,3), activation=('relu')))
    CNN.add(MaxPool2D(pool_size=(2,2)))
    CNN.add(Conv2D(64, (3,3), activation=("relu")))
    CNN.add(MaxPool2D(pool_size=(2,2)))
    
    #comands to make longer model. 
    if inc_2ndbatch:
        CNN.add(Conv2D(64, (3,3), activation=("relu")))
        CNN.add(MaxPool2D(pool_size=(2,2)))
        CNN.add(Conv2D(64, (3,3), activation=("relu")))
        CNN.add(MaxPool2D(pool_size=(2,2)))
    if inc_neck:
        CNN.add(Conv2D(64, (3,3), activation=("relu")))
        CNN.add(Conv2D(64, (3,3), activation=("relu")))


    CNN.add(Flatten())
    # CNN.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    
    CNN.add(Dense(2, activation=('softmax'), kernel_initializer='normal')) 
    
    # # Section 2: a
    loss = loss_options(loss_type, alpha, gamma)
    
    # #compile based on loss function from 2a
    CNN.compile(loss=loss, optimizer=optimizers.Adam(learning_rate=lr),
                metrics=["accuracy"])
   
    # # section 2: b
    # #call back for learning rate decay
    callback = LR_options(lr,lr_type, epochs)

    # # section 2: C
    es = MarochovCallback(threshold=0.90)
    
    if callback == None:
        callback = [es]
    else:
        callback = callback+[es]
    
    CNN.fit(Xy_train, epochs=epochs, validation_data=(Xy_test)) #, callbacks=(callback) )

    gc.collect()

        
    return CNN

### Section Two: A
def loss_options(loss_type, alpha, gamma):
    if loss_type == 'focal':
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
        
    else:
        loss = "sparse_categorical_crossentropy"
    return loss


### Section 2: B
def LR_options(lr, lr_type, epochs):
    
    if lr_type == 'plain':
        callback = None   
    else:
        #decay type
        initial_learning_rate = lr
        if lr_type == 'time':
            
            decay = lr / epochs
            def lr_time_based_decay(epoch, lr):
                return initial_learning_rate * 1 / (1 + decay * epoch)
            decay_func = lr_time_based_decay
            
        elif lr_type == 'step':
            def lr_step_decay(epoch, lr):
                drop_rate = 0.5
                epochs_drop = 4.0
                return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
            decay_func = lr_step_decay
        else:
            def lr_exp_decay(epoch, lr):
                k = 0.1
                return initial_learning_rate * math.exp(-k*epoch)
            decay_func = lr_exp_decay
        print(decay_func)
        callback = [LearningRateScheduler(decay_func, verbose=1)]
    return callback

### Section 2: C
class MarochovCallback(callbacks.Callback):
    def __init__(self, threshold):
        super(MarochovCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_accuracy"]
        if accuracy >= self.threshold:
            print('')
            print('Validation accuracy target reached, stopping training')
            print('')
            self.model.stop_training = True
###############################################################################
#%% Section 3
###############################################################################

def saveModelAndOutputs(model, saveModel, outfile):
    
    
    # put in the model output section
    parentDir = 'D:/Code/RiverTwin/ZZ_ModelOutputs'
    path = os.path.join(parentDir, outfile)
    
    try:
        #make an outputs folder
        os.mkdir(path)
    except FileExistsError:
        # if the file exists just make the folder name different by adding an X
        new_output = outfile + 'X'
        # and repeat
        path = os.path.join(parentDir, new_output)
        os.mkdir(path)

        
    
    # Section 3: A
    graphs(model, saveModel, outfile, path)
    
    #Section 3: B
    saveWholeModel(model, path)
    
    #Section 3:C
    saveTrainingEpochs(model, path)
#%%
### Section 3:A
def graphs(model, saveModel, outfile, path):
    
    ### loss
    plt.figure()
    data = model.history.history
    plt.title(outfile+' Loss')
    plt.plot(data['loss'], label='Loss')
    plt.plot(data['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    if saveModel: 
        name = 'loss.png'
        name = os.path.join(path, name)
        plt.savefig(name, dpi=600)
        
    ### Accuracy
    plt.figure()
    data = model.history.history
    plt.title(outfile+' Accuracy')
    plt.plot(data['accuracy'], label='Accuracy')
    plt.plot(data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    if saveModel: 
        name = 'accuracy.png'
        name = os.path.join(path, name)
        plt.savefig(name, dpi=600)
    
    
### Section 3:B        
def saveWholeModel(model, path):
    name = 'model'
    name = os.path.join(path, name)
    
    #check on first run for file extension
    model.save(name)

    
    
### Section 3:C
def saveTrainingEpochs(model, path):
    hd = model.history.history
    df = pd.DataFrame.from_dict(hd)
    name = 'trainingEpochs.csv'
    name = os.path.join(path, name)
    
    df.to_csv(name)

        
        

    
##############################################################################
def GPU_SETUP():
    '''setup for RTX use of mixed precision
    directly from Carbonneu '''
    #Needs Tensorflow 2.4 and an RTX GPU
    if ('RTX' in os.popen('nvidia-smi -L').read()) and ('2.10' in tf.__version__):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
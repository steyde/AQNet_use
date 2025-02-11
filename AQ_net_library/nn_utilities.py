# Importing all required libraries
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
import random
import os
import scipy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics
from scipy import interpolate #for fill nan
import itertools
from mat73 import loadmat as loadmat73
from scipy.io import loadmat 
from scipy.interpolate import PchipInterpolator
import pandas as pd
import pickle
import scipy.signal
from scipy.signal import decimate
from pandas import Series
from sklearn.impute import KNNImputer
# Imports for the legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from keras.callbacks import Callback

from scipy.interpolate import PchipInterpolator, interp1d

#tf.config.optimizer.set_experimental_options({"xla": False})

def plotPSD(input_signal, fs):

    input_signal = input_signal - np.mean(input_signal)    
    freq, psd = scipy.signal.welch(input_signal, nfft = fs*60, fs=fs, scaling='density')
    plt.figure()
    plt.plot(freq, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Unilateral Power Spectral Density')
    plt.show()
    return

def ChopSignals(X, y, block_length, shift=60, data_shift=False):
    
    """
    DESCRIPTION
    This function is used to chop into various fragments the FHR trace and the annotations
    
    INPUT
    - X: the FHR traces arrays
    - y: the annotations arrays
    - block_length: the length of the block into which to chop the arrays (in samples)
    - shift (int, default=20): the shift in samples of the block
    - data_shift (bool, default=False): whether to use a sliding window or a fixed window approach
    
    OUTPUT
    - chopped_signals_list, chopped_annotations_list: the chopped X and y lists
    """
    
    # Initialization of the final output lists
    chopped_signals_list = list()
    chopped_annotations_list = list()
    
    # Iteratively extracting the signals
    for i in range(len(X)):
        
        # Retriving signal and annotations
        signal = X[i]
        annotation = y[i]

        # Precomputing the signal length and the annotation length
        signal_length = len(signal)
        annotation_length = len(annotation)
        
        # Computing the resolution of the downsampling
        resolution = signal_length/annotation_length
        
        # If I do not want to use overlapping windows, then I extract the signal block by block
        if data_shift == False:
            
            # Computing the number of integer segments (Extra segments of less than block_length minutes will be discarded)
            number_of_segments = np.floor(signal_length/block_length).astype("int")
    
            # Iteratively chopping the signal and the annotations and appending them to the list
            for j in range(number_of_segments):
                # Start and end of the segment to chop
                start = 0 + j*block_length 
                end = block_length + j*block_length
                
                # Sanity check
                if j==1:
                    assert start % resolution == 0
                    assert end % resolution == 0
                
                # Chopping 
                chopped_signal = signal[start:end]
                chopped_annotation = annotation[int(start/resolution):int(end/resolution)]
                # Appending to the lists
                chopped_signals_list.append(chopped_signal)
                chopped_annotations_list.append(chopped_annotation)
        
        # Otherwise I want to use overlapping windows
        else:
            j = 0
            while (j*shift + block_length) < signal_length:
                # Start and end of the segment to chop
                start = j*shift
                end = j*shift + block_length
                
                # Sanity check
                if j==1:
                    assert start % resolution == 0
                    assert end % resolution == 0
                
                # Chopping 
                chopped_signal = signal[start:end]
                chopped_annotation = annotation[int(start/resolution):int(end/resolution)]
                # Appending to the lists
                chopped_signals_list.append(chopped_signal.reshape(-1))
                chopped_annotations_list.append(chopped_annotation.reshape(-1))
                # Moving window
                j = j+1
            
    # Converting to a numpy array now that they have the same length
    chopped_signals_list = np.array(chopped_signals_list)
    chopped_annotations_list = np.array(chopped_annotations_list)
    
    # Expanding the final dimension so that it is compatible with Keras
    # From (n_observations, n_samples) to (n_observations, n_samples, 1)
    chopped_signals_list = np.expand_dims(chopped_signals_list, 2)
    chopped_annotations_list = np.expand_dims(chopped_annotations_list, 2)
    
    # Returning signals and annotations lists
    return chopped_signals_list, chopped_annotations_list

def FindStartStopStates(state_array):
    
    """
    DESCRIPTION
    This function is used to find the [start, stop] values of a particular sequence of state
    
    INPUT
    - state_array: the array which encodes the states
    
    OUTPUT
    - states_dict: a dictionary whose keys are the unique encoded states and whose values 
                   are the [start, stop] values found for that state
    """

    from itertools import groupby

    #GIULIO 01/11/2023
    # I want to make it work when state_array is ['T','T','T','T','A','A','A','A','A' ...]
    if isinstance(state_array,list):
        #from letter to number
        state_array = [1 if item == 'A' else 0 if item == 'Q' else -1 if item == 'T' else -2 for item in state_array]
        #from list to np.array
        state_array = np.array([int(item) for item in state_array])

    # Retrieve the unique states predicted by the model
    states = np.unique(state_array)

    # I create a dictionary of the states which will contain the start and stop values for that particular state
    states_dict = dict()

    # Cycling on the states
    for s in states:

        # I create a mask of the state
        mask = (state_array == s)

        # Finding sequences and lengths
        # seqs = [(key, length), ...]
        seqs = [(key, len(list(val))) for key, val in groupby(mask)]
        # Finding start positions of sequences
        # seqs = [(key, start, length), ...]
        seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]

        # Sequences start and stop
        states_dict[s] = [[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == 1]
        
    # Returning the dictionary of states which encode the start and stop values of the sequences
    return states_dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def PlotResult(signal, prediction, softmax, fs=2, title="Signal Segmentation", plotSoftmax=1):
    """
    DESCRIPTION
    This function is used to plot the results of the segmentation
    
    INPUT
    - signal: the FHR signal
    - prediction: the array which encodes the segmentation into the different states 
                  (if prediction==None, then it does not plot the states)
    - title (default="Signal Segmentation"): the title of the figure

    OUTPUT
    - None: it plots a figure
    """
    
    # Resample predictions if fs > 2
    if fs > 2:
        prediction = ResampleLabel(prediction, 2, fs)
        
    # Plotting the result
    if plotSoftmax:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))

    # Font size update
    plt.rcParams.update({'font.size': 14})

    # Axis and title label
    ax1.set_title(title)
    ax1.set_ylabel("FHR [bpm]")
    ax1.set_xlabel("Time [s]")
    
    # Initializing legend with just the FHR signal
    legend_elements = [Line2D([0], [0], color='#000', lw=1, label='FHR')]

    # Changing the x axis from samples to seconds by converting it using the sampling frequency fs
    start = 0
    stop = len(signal) / fs
    step = 1 / fs
    t = np.arange(start, stop, step)

    if prediction is not None:
        # Plot the proposed states of the model as background color
        states_dict = FindStartStopStates(prediction)
        # Quiet (blue)
        if 0 in states_dict.keys():
            for [start, stop] in states_dict[0]:
                ax1.axvspan(start / fs, stop / fs, facecolor='#0000FF', alpha=0.25)
        # Activity (red)
        if 1 in states_dict.keys():
            for [start, stop] in states_dict[1]:
                ax1.axvspan(start / fs, stop / fs, facecolor='#FF0000', alpha=0.25)

        # Adding the legend of the states
        legend_elements.append(Patch(facecolor='#0000FF', edgecolor='k', label='Quiet'))
        legend_elements.append(Patch(facecolor='#FF0000', edgecolor='k', label='Active'))
        
        # Transition (gray)
        if -1 in states_dict.keys():
            legend_elements.append(Patch(facecolor='#C2BF9B', edgecolor='k', label='Undetermined'))
            for [start, stop] in states_dict[-1]:
                ax1.axvspan(start / fs, stop / fs, facecolor='#C2BF9B', alpha=0.25)
        # Interpolated (white)
        if -2 in states_dict.keys():
            for [start, stop] in states_dict[-2]:
                ax1.axvspan(start / fs, stop / fs, facecolor='#FFFFFF', alpha=0.25)
        
    # Plot the FHR signal
    ax1.plot(t, signal, color="#000", lw=1)

    # Custom legend plot
    ax1.legend(handles=legend_elements, loc='lower right')

    # Limits of the x and y axis
    ax1.set_ylim([60, 200])  # bpm
    ax1.set_xlim([0, t[-1]])

    # Make the grid more dense
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    if plotSoftmax:
        # Softmax predictions
        start = 0
        stop = len(signal) / 2
        step = 1 / 2
        t = np.arange(start, stop, step)
        ax2.plot(t, softmax, lw=1)
        ax2.set_ylim([0, 1])
        ax2.axhline(0.5, linestyle='--', linewidth=0.5)
        # Axis and title label
        ax2.set_title("Probability of Active State")
        ax2.set_xlabel("Time [s]")
        ax2.set_xlim([0, t[-1]])

        # Make the grid more dense
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    # Font size returning to default
    plt.rcParams.update({'font.size': 10})


    
    
def FlattenList(_2d_list):
    
    """
    DESCRIPTION
    This function flattens a 2d list
    
    INPUT:
    - _2d_list = the 2D list to be flattened
    
    OUTPUT:
    - flat_list: the flattened list
    """
    
    flat_list = []
    
    # Iterate through the outer list
    for element in _2d_list:
        # If the element is of type list or np.ndarray
        if type(element) is list or type(element) is np.ndarray:
            # Iterating through the sublist
            for item in element:
                flat_list.append(item)
        # Else append the single element
        else:
            flat_list.append(element)
            
    return flat_list

def PredictAQ(signal, model, fs, units, Transition = False, Threshold = -1, MinLen = 6, smooth=True, sigma = 0.5, scaling_factor = [140.03507895,  11.64249628]):

    '''
    12/10/2023 Politecnico di Milano

    Include basic pre-processing of the FHR signal and inputs before calling the network
    
    '''
    #sanity check
    if units!='bpm' and units!= 'ms':
        raise Exception("units either in bpm or ms")
    if fs not in [-1,2,4,8]:
        raise Exception("fs can be 2,4, 8 or -1 (RR series)")
    
    MinLen = int(MinLen)
    signal = signal.astype(float)
    sigma = int(120*sigma)          #transform in samples

    #if Threshold has not been specified, set defaul values:
    if Threshold[0] == -1:
        if Transition:
            Threshold = [0.4, 0.6]
        else:
            Threshold = 0.5

    # preprocess the signal (converto to bpm sampled @2Hz)
    signal,invalid_samples = PreProcess(signal, fs, units)
    
    # Apply the network
    predictions, softmax_prediction_list = UseNet(signal, model, sigma, Transition, Threshold, MinLen, invalid_samples, smooth, verbose = False, scaling_factor = scaling_factor)
    
    return signal, predictions, softmax_prediction_list
    

def UseNet(signal, model, sigma, Transition,  Threshold, MinLen, invalid_samples, smooth=True, verbose=False, scaling_factor = [140.03507895,  11.64249628]):

    '''
    Chop the preprocessed FHR signal into windows of 2400 samples (20 minutes) and apply the network, 
    then realign and post-process the output to obtain the final segmentation

    '''
    original_signal_length = len(signal)

    #Rescaling
    if len(scaling_factor)==2:
        signal = (signal-scaling_factor[0])/scaling_factor[1]
    
    #Sampling frequency of the network
    fs = 2 #Hz
    
    #length of input to the U-net-like network DO NOT CHANGE
    block_length = 20*60*fs
    
    _,softmax_prediction_list = Predict_U_Net(signal, model, block_length, fs, smooth=smooth, verbose=0, nn_output="softmax")
      
    # Processing the softmax predictions
    predictions = PostProcessPred(softmax_prediction_list,smooth,Transition,invalid_samples,Threshold,sigma,MinLen)
        
    # Returning the Neural Network predictions
    return predictions[:original_signal_length], softmax_prediction_list[:original_signal_length]

def PreProcess(signal,fs,units):
    
    '''
    preprocess the FHR signal to:
    - identify and (possibly) remove invalid samples
    - make the input signal compatible with the network (resample to 2Hz and invert to bpm)

    '''
    
    #added 14/11
    #return also invalid samples (interpolated) which will be discarded in post processing

    #initialize to vector of zero
    invalid = np.zeros_like(signal)
    
    # STEP 0: interpolate missing values (works for missing values == 0 or NaN)
    invalid[signal==0] = 1 #add (likely) maternal?
    signal[signal==0] = np.nan
    signal = fill_nan(signal) #forse è meglio spline
    signal = hampel(signal)# Hampel filter

    if np.isnan(signal).any():
        #debug
        print('Nan in signal!!')
    
    
    #STEP 1: resample rr series
    if fs==-1: #RR series
        #assume that RR series are in ms
        if units=='bpm':
            signal = 60000/signal
            units = 'ms'
            
        #interpolate to 8Hz (to be sure to respect Shannon Theoreme)
        t_original = np.cumsum(signal)
        
        #set first value to 0
        t_original -= t_original[0] 
        
        #interoplating function
        f = PchipInterpolator(t_original, signal)
        
        # sample rate for interpolation
        fs = 8.0
        step = 1000 / fs # Assuming step in ms

        # now we can sample from interpolation function
        t = np.arange(0, np.max(t_original)+step, step)
        signal = f(t)

        #also the intepolatates
        F_InterpP = PchipInterpolator(t_original, invalid)
        invalid  = np.round(F_InterpP(t))

        #t to second
        t = t/1000
        
    # STEP 2: downsample to 2Hz
    if fs > 2:        
        fs_new = 2
        signal = decimate(signal, int(fs/fs_new))
        invalid = invalid[::int(fs/fs_new)]
        fs = fs_new
        
    #STEP 3
    #convert to bpm
    if units=='ms':
        signal = 60000/signal
        
    return signal,invalid
 

def hampel(vals_orig, k=15, t0=4):
    
    '''
    Apply hample filter to remove outliers
  
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    
    # Convert to pandas series
    vals_orig = Series(vals_orig)
    
    #Make copy so original not edited
    NanValues = np.isnan(vals_orig)
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    
    vals.interpolate(method='linear')
    
    vals.ffill(inplace=True)
    vals.bfill(inplace=True)

    vals[NanValues] = np.nan
    
    vals = np.array(vals)

        
    return(vals)


def fill_nan(A, type="linear"):
    """
    DESCRIPTION
        This function is used to fill the nan values in the array A
    INPUT
        - A: the array in which to fill the nan values
        - type (default = "linear"): the type of interpolation to use (either "linear" or "pchip")
    OUTPUT    
        - B: the array with the nan values filled
    """
    A = np.asarray(A, dtype=np.float64)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))

    # Linear interpolation
    if type == "linear":
        f = interp1d(inds[good], A[good], bounds_error=False)
    # Pchip Interpolation
    elif type == "pchip":
        f = PchipInterpolator(inds[good], A[good], extrapolate=False) # I do not extrapolate
    else:
        raise Exception("Type of interpolation not recognized (either 'linear' or 'pchip')")

    B = np.where(np.isfinite(A),A,f(inds))
 
    # I take care of the first or last segments in which I could have nan values
    ind = np.where(~np.isnan(B))[0]
    first, last = ind[0], ind[-1]
    B[:first] = B[first]
    B[last + 1:] = B[last]
 
    return B

def ResampleLabel(downsampled_labels, f_original, f_target):
    
    """
    DESCRIPTION
    This function is used to resample the labels
    
    INPUT
    - downsampled_labels: the associated downsampled segmentation label
    - fs: the original sampling frequency (in Hz) of the network (2)
    - f target: the new frequency resolution (e.g., 4Hz)
    
    OUTPUT
    - resampled_labels: the resampled labels which match the original fs
    """
    
    # Test upsampling to see if they match
    upsampled_labels = list()
    for l in downsampled_labels:
        upsampled_labels.append([l for i in range(int(f_target/f_original))])

    if type(downsampled_labels[0]) == int or float:
        upsampled_labels = np.array(upsampled_labels).flatten()
    else:
        upsampled_labels = FlattenList(downsampled_labels)


    return upsampled_labels


def PostProcessPred(softmax_prediction_list,smooth,Transition,invalid_samples,Threshold, sigma, MinLen):

    '''
    Post Process the predictions of the network to obtain the final segmentation
     -> smooth the softmax predictions to avoid fust transitions
     -> impose a minimum length to states

    '''
    
    if smooth:
        # Applying the gaussian filter
        softmax_prediction_list = gaussian_filter1d(softmax_prediction_list, sigma, mode="reflect") #change kernel to account for temporality

    if Transition:
        predictions = PostProcessTransition3(softmax_prediction_list,Threshold,MinLen)    
    else:
        # Thresholding the softmax predictions   
        predictions = PostProcessTransition2(softmax_prediction_list,Threshold,MinLen) 

    if False:
        #THIS PART MUST BE CHECKED

        #remove interpolated parts from prediction, but only when the blocks of interpolated points are longer than 30 seconds
        # Use itertools.groupby to group consecutive elements
        grouped_blocks = [list(g) for k, g in itertools.groupby(invalid_samples)]
        # Filter blocks longer than 60 samples
        filtered_blocks = [block if (block[0] == 0 or len(block) > 60) else [0] * len(block) for block in grouped_blocks]

        # Concatenate the selected blocks to get the final result
        invalid_blocks = np.concatenate(filtered_blocks) 
        predictions = ['I' if value == 1 else element for value, element in zip(invalid_blocks, predictions)]
    
    return predictions


def PostProcessTransition2(softmax_prediction_list,Threshold,MinLen):

    '''
    Called by PostProcessPred whene there are only 2 possible states (Activity and Quiet)

    '''
    
    #create list of symbols
    predictions = ["Q" if val < Threshold else "A" for val in softmax_prediction_list]
    
    #a lot of this is a copy of PostProcessTransition3 @TODO: merge the two functions

    #divide in blocks
    blocks_1 = divide_list_into_blocks(predictions) #all segments are contained in blocks_1

    #initialize processed
    blocks_2 = [None] * len(blocks_1) #blocks_2 = contains only "valid" segments

    #first round: assign valid blocks (> MinLen, in samples)
    for index in range(0,len(blocks_1)):
        item = blocks_1[index]
        if len(item)>MinLen*120:
            blocks_2[index] = item

    #for the extremes consider valid Half of Min Len
    #beginning
    HalfWin = int((MinLen/2))
    if len(blocks_1[0])>(120*HalfWin):
        blocks_2[0] = blocks_1[0]
    elif blocks_2[1] != None:
        blocks_2[0] = [blocks_2[1][0] for _ in range(len(blocks_1[0]))]
    else:
        blocks_2[0] = blocks_1[0]
    #end
    if len(blocks_1[-1])>(120*HalfWin):
        blocks_2[-1] = blocks_1[-1] 
    elif blocks_2[-2] != None:
        blocks_2[-1] = [blocks_2[-2][0] for _ in range(len(blocks_1[-1]))]
    else:
        blocks_2[-1] = blocks_1[-1]

    # here I assign short states to the other state ONLY IF they are "surrounded" by valid states. 
    # Otherwise, the network is not sure about the state and I cannot confidently assign any state
    for index in range(1,len(blocks_1)-1):
        item = blocks_1[index]
        if len(item)<=MinLen*120:
            if ((blocks_2[index-1]) is None) or ((blocks_2[index+1]) is None): #has not been assigned a valid
                #qua si potrebbe inventarsi qualcosa, intanto il problema rimane:
                blocks_2[index] = blocks_1[index]
            else: #se sono due i possibili stati per forza sono uguali gli estremi!
                blocks_2[index] = [blocks_2[index-1][0] for _ in range(len(item))]

    #collapse consecutive states that are the same
    blocks_3 = list()
    blocks_3.append(blocks_2[0])
    index = 1
    while(index < len(blocks_2)):
        new_item = blocks_2[index]
        old_item = blocks_3[-1]
        if new_item[0] == old_item[0]:
            blocks_3[-1] = [new_item[0] for i in range(len(new_item)+len(old_item))]
        else:
            blocks_3.append(new_item)
        index += 1

    for index in range(1,len(blocks_3)-1):
        item = blocks_3[index]
        if (len(item)<=MinLen*120) and (blocks_3[index-1][0]==blocks_3[index+1][0]):
            blocks_3[index] = [blocks_3[index-1][0] for _ in range(len(item))]

    #reshape to a list of symbols
    predictions = [item for sublist in blocks_3 for item in sublist]

    return predictions



def PostProcessTransition3(softmax_prediction_list,Threshold,MinLen):

    '''
    Called by PostProcessPred whene there are only 3 possible states (Activity, Quiet and Transition/Undefined)

    Basically the same as PostProcessTransition2, but with 3 states

    '''
    
    #create list of symbols
    predictions = ["Q" if val < Threshold[0] else "T" if val < Threshold[1] else "A" for val in softmax_prediction_list]

    #divide in blocks
    blocks_1 = divide_list_into_blocks(predictions)#raw
    
    #initialize processed
    blocks_2 = [None] * len(blocks_1)
    
    #first round: assign valid blocks (>6 min)
    for index in range(0,len(blocks_1)):
        item = blocks_1[index]
        if len(item)>MinLen*120:
            blocks_2[index] = item

    #for the extremes consider valid 3 minutes
    # Beginning 
    HalfWin = int((MinLen/2))
    if len(blocks_1[0])<(120*HalfWin):
        if blocks_1[0][0]=='T' and (blocks_2[1] != None):
            blocks_2[0] = [blocks_2[1][0] for _ in range(len(blocks_1[0]))]
        else:
            blocks_2[0] = ['T' for _ in range(len(blocks_1[0]))] #set to transition  
    else:
        blocks_2[0] = blocks_1[0]
    #end
    if len(blocks_1[-1])<(120*HalfWin):
        if blocks_1[-1][0]=='T' and (blocks_2[-2] != None):
            blocks_2[-1] = [blocks_2[-2][0] for _ in range(len(blocks_1[-1]))]
        else:
            blocks_2[-1] = ['T' for _ in range(len(blocks_1[-1]))] #set to transition
    else:
        blocks_2[-1] = blocks_1[-1] 
     
    #now iterate central part for the others
    for index in range(1,len(blocks_1)-1):
        item = blocks_1[index]
        if len(item)<=MinLen*120:
            surrounded = 1
            if ((blocks_2[index-1]) is None) or ((blocks_2[index+1]) is None): #has not been assigned a valid
                surrounded = 0
            elif blocks_2[index-1][0]!=blocks_2[index+1][0]:
                surrounded = 0
            if surrounded: #circondato
                blocks_2[index] = [blocks_2[index-1][0] for _ in range(len(item))]
            else: #qua si potrebbe rendere più sofisticata la cosa
                blocks_2[index] = ['T' for _ in range(len(item))]

    #there could be short T states surrounded by A or Q:
    #1-collapse consecutive states that are the same
    #2-run trough the list and remove T states < 6min

    blocks_3 = list()
    blocks_3.append(blocks_2[0])
    index = 1
    while(index < len(blocks_2)):
        new_item = blocks_2[index]
        old_item = blocks_3[-1]
        if new_item[0] == old_item[0]:
            blocks_3[-1] = [new_item[0] for i in range(len(new_item)+len(old_item))]
        else:
            blocks_3.append(new_item)
        index += 1

    for index in range(1,len(blocks_3)-1):
        item = blocks_3[index]
        if (len(item)<=6*MinLen) and (blocks_3[index-1][0]==blocks_3[index+1][0]):
            blocks_3[index] = [blocks_3[index-1][0] for _ in range(len(item))]

    predictions = [item for sublist in blocks_3 for item in sublist]

    return predictions

def divide_list_into_blocks(input_list):
    blocks = []
    current_block = []

    for item in input_list:
        if not current_block or item == current_block[0]:
            current_block.append(item)
        else:
            blocks.append(current_block)
            current_block = [item]

    if current_block:
        blocks.append(current_block)

    return blocks

#12/07
def DeleteSpikes(signal, unit = 'bpm', kernel = 41, Threshold = 25/2):
    
    if unit == 'ms':
        signal = 60000/signal

    baseline = scipy.signal.medfilt(signal, kernel_size = kernel)
    invalid = np.abs(signal - baseline) > Threshold

    signal[invalid] = baseline[invalid]

    if unit == 'ms':
        signal = 60000/signal

    return signal


def DeleteSpikes2(signal, Threshold = 30):

    values_to_be_replaced = np.abs(np.diff(signal))>Threshold
    values_to_be_replaced = expand_boolean_array(values_to_be_replaced, expand_by=1)
    values_to_be_replaced = np.append(values_to_be_replaced, 0).astype("bool") # Add the last value due to the difference operation
        
    smoothed_signal = scipy.signal.medfilt(signal, 21)
    corrected_signal = np.zeros(len(signal))
    for i, is_to_replace in enumerate(values_to_be_replaced):
        if is_to_replace==True:
            corrected_signal[i] = smoothed_signal[i]
        else:
            corrected_signal[i] = signal[i]
    
    return corrected_signal

def expand_boolean_array(arr, expand_by):
    """
    Expands the True values in a boolean array by a given amount.
   
    Parameters:
    arr (list or np.array): Input boolean array.
    expand_by (int): Number of cells to expand True values.
   
    Returns:
    np.array: Expanded boolean array.
    """
    # Convert to numpy array for easy manipulation
    arr = np.array(arr)
   
    # Pad the array to avoid boundary issues
    padded_arr = np.pad(arr, pad_width=expand_by, mode='constant', constant_values=0)
   
    # Create a structure for dilation (a window of size 2*expand_by+1)
    structure = np.ones(2 * expand_by + 1)
   
    # Dilate the array using the structure
    expanded_arr = np.convolve(padded_arr, structure, mode='same')
   
    # Clip the result to 0 or 1
    expanded_arr = np.clip(expanded_arr, 0, 1)
   
    # Remove the padding
    expanded_arr = expanded_arr[expand_by:-expand_by]
   
    return expanded_arr

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]    
    return ret[n - 1:] / n 

def GenerateReport(prediction):
    blocks = []
    start_index = 0
    current_symbol = prediction[0]

    for i in range(1, len(prediction)):
        if prediction[i] != current_symbol:
            blocks.append((current_symbol, start_index, i - 1))
            current_symbol = prediction[i]
            start_index = i

    # Append the last block
    blocks.append((current_symbol, start_index, len(prediction) - 1))

    return blocks


def custom_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        bce = -bce

        # Apply the weights to each class individually
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_bce = weight_vector * bce

        # Return the mean error
        return tf.reduce_mean(weighted_bce)

    return weighted_binary_crossentropy

def Predict_U_Net(signal, model, block_length, fs, smooth=True, verbose=0, nn_output="softmax"):
    """
    DESCRIPTION
    This function applies the neural network model to predict active and quiet periods from the FHR time series
    with 50% overlap between windows.

    INPUT
    - signal: the FHR signal which has been rescaled
    - model: the U-Net neural network model
    - block_length: the input length to the U-Net
    - smooth (default = True): whether or not to apply a gaussian smoothing to post-process the predictions
    - nn_output (default="softmax"): the output of the neural network (either "softmax" or "sigmoid")

    OUTPUT
    - predictions: the U-Net predictions
    - averaged_predictions: the smoothed predictions before thresholding

    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Computing the length of the signal
    signal_length = len(signal)

    if signal_length < block_length:
        #extend the signal by reflecting the last part up to block_length
        padSize = block_length - signal_length
        signal = np.concatenate((signal, np.flip(signal[-padSize:], axis=0))) 
        CutSignal = True
        signal_length = len(signal)
    else:
        CutSignal = False

    # Define the step size for 66% overlap
    step_size = block_length // 3

    # Compute the number of windows with 50% overlap
    number_of_segments = (signal_length - block_length) // step_size + 1

    # Initialize an array to hold cumulative predictions and a counter for averaging
    cumulative_predictions = np.zeros(signal_length)
    count = np.zeros(signal_length)

    # Process each window
    for j in range(number_of_segments):
        # Start and end of the segment
        start = j * step_size
        end = start + block_length

        # Extract the segment
        chopped_signal = signal[start:end]

        chopped_signal = np.array(chopped_signal, dtype=np.float64)
        chopped_signal = chopped_signal.reshape(1, -1)

        # Make the prediction for the current segment
        softmax_prediction = model.predict(chopped_signal, verbose=verbose)
        if nn_output == "softmax":
            softmax_prediction = softmax_prediction[0][:, 0].flatten()  # Keep only the first output neuron
        elif nn_output == "sigmoid":
            softmax_prediction = softmax_prediction.flatten()  # Only one output neuron
        else:
            raise ValueError("Invalid nn_output. Choose 'softmax' or 'sigmoid'.")

        # Add the predictions to the cumulative array
        cumulative_predictions[start:end] += softmax_prediction
        count[start:end] += 1

    # Handle the last segment if needed
    remainder = (signal_length - block_length) % step_size
    if remainder > 0:
        start = signal_length - block_length
        end = signal_length

        chopped_signal = signal[start:end]
        chopped_signal = np.array(chopped_signal, dtype=np.float64)
        chopped_signal = chopped_signal.reshape(1, -1)
            
        softmax_prediction = model.predict(chopped_signal, verbose=verbose)
        if nn_output == "softmax":
            softmax_prediction = softmax_prediction[0][:, 0].flatten()
        elif nn_output == "sigmoid":
            softmax_prediction = softmax_prediction.flatten()

        cumulative_predictions[start:end] += softmax_prediction
        count[start:end] += 1

    # Avoid division by zero
    count[count == 0] = 1

    # Average the overlapping predictions
    averaged_predictions = cumulative_predictions / count

    # Apply smoothing if required
    if smooth:
        sigma = 60 * fs  # Standard deviation for Gaussian filter (1 minute)
        averaged_predictions = gaussian_filter1d(averaged_predictions, sigma, mode="reflect")

    #If the signal was shorter thank block length and has been extended
    if CutSignal:
        averaged_predictions = averaged_predictions[:signal_length]

    # Threshold the predictions
    predictions = np.round(averaged_predictions)

    return predictions, averaged_predictions

def compute_scores(y_true, y_pred):

    # Removing nan indexes
    y_true_nan_indexes = np.isnan(y_true)
    y_pred_nan_indexes = np.isnan(y_pred)
    nan_indexes = np.logical_or(y_true_nan_indexes, y_pred_nan_indexes)
    y_true = y_true[~nan_indexes]
    y_pred = y_pred[~nan_indexes]

    # Computing the scores
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    F1 = metrics.f1_score(y_true, y_pred)
    macro_F1 = metrics.f1_score(y_true, y_pred, average="macro")
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, normalize="true").ravel()

    return MCC, accuracy, balanced_accuracy, F1, macro_F1, kappa, tn, fp, fn, tp



def mcc_metric_keras(y_true, y_pred):

    """
    DESCRIPTION:
        This compute the Matthews Correlation Coefficient (MCC) between the true and predicted annotations array (Keras implementation)
    INPUT:
        - y_true: the binary true annotation array from the clinician
        - y_pred: the binary predicted annotation array
    OUTPUT:
        - mcc_score: the computed MCC score
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    tn = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * y_pred, 0, 1)))
    fn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * (1 - y_pred), 0, 1)))

    # Scaling to avoid overflow
    #scale = np.max([tp, tn, fp, fn])
    #tp = tp / scale
    #tn = tn / scale
    #fp = fp / scale
    #fn = fn / scale
    scale = 1 # We do not need to scale the values since the timeseries are small

    # Numerator and denominator of MCC
    num = tp * tn - fp * fn
    den = tf.sqrt(tp + fp) * tf.sqrt(tp + fn) * tf.sqrt(tn + fp) * tf.sqrt(tn + fn)
    
    # Total number of elements in y_true
    total_elements = tf.cast(tf.size(y_true), tf.float32)

    # Compute MCC
    mcc_score = tf.where(
        tf.logical_and(num != 0, den != 0),  # Case 1: Standard MCC
        num / (den + tf.keras.backend.epsilon()),
        tf.zeros_like(num)  # Case 2: Indeterminate (0/0)
    )

    # Handle case 3: Perfect agreement (tp = total_elements)
    mcc_score = tf.where(
        tf.equal(tp, total_elements),
        tf.ones_like(mcc_score),
        mcc_score
    )

    # Handle case 4: No agreement (fp = total_elements)
    mcc_score = tf.where(
        tf.equal(fp, total_elements),
        -tf.ones_like(mcc_score),
        mcc_score
    )

    return mcc_score


#https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_metric(y_true, y_pred):
    
    """
    DESCRIPTION
    This function is used to compute the recall metric for the neural network
    
    INPUT
    - y_true: the true  labels
    - y_pred: the predicted labels
    
    OUTPUT
    - recall: the recall
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_metric(y_true, y_pred):
    
    """
    DESCRIPTION
    This function is used to compute the precision metric for the neural network
    
    INPUT
    - y_true: the true labels
    - y_pred: the predicted labels
    
    OUTPUT
    - precision: the precision
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_metric(y_true, y_pred):
    
    """
    DESCRIPTION
    This function is used to compute the f1-score metric for the neural network
    
    INPUT
    - y_true: the true labels
    - y_pred: the predicted labels
    
    OUTPUT
    - f1: the f1 score
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1_score

def macro_f1_metric_keras(y_true, y_pred):
    
    """
    DESCRIPTION
    This function is used to compute the macro f1-score metric for the neural network
    
    INPUT
    - y_true: the true labels
    - y_pred: the predicted labels
    
    OUTPUT
    - macro_f1_score: the macro f1 score
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Class 1
    f1_first = f1_metric(y_true, y_pred)
    
    # Class 0 (I invert the labels and compute again the score)
    y_true_inverted = 1 - y_true
    y_pred_inverted = 1 - y_pred
    f1_second = f1_metric(y_true_inverted, y_pred_inverted)
    
    # Compute macro F1-score as the unweighted mean
    macro_f1_score = 0.5*(f1_first+f1_second)
    
    return macro_f1_score



def ReframeLabelsProbability(y):
    """
    DESCRIPTION
        This function reframes the labels into a probability format.
    INPUT
        - y: the labels (list of arrays/lists)
    OUTPUT
        - final_labels: the reframed labels (NumPy array)
    """
    # Reframe each signal's labels in a vectorized way
    final_labels = [
        np.stack([np.ravel(signal), 1 - np.ravel(signal)], axis=1)
        for signal in y
    ]
    # Convert to NumPy array
    return np.array(final_labels)

#use this callback to evaluate the network using whole signals instead of chuncks in validation
# Callback to store predictions and compute F1 score
class EvaluationCallback(Callback):
    def __init__(self, model, X_val_fold, y_val_fold, block_length, fs):
        super().__init__()
        self.model = model
        self.X_val_fold = X_val_fold
        self.y_val_fold = y_val_fold
        self.block_length = block_length
        self.fs = fs
        self.fold_pred = []
        self.fold_true = []
        self.macro_F1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        self.fold_pred.clear()
        self.fold_true.clear()
        for i in range(len(self.X_val_fold)):
            signal = self.X_val_fold[i]
            y_pred_i, _ = Predict_U_Net(
                signal, self.model, self.block_length, self.fs, smooth=False, nn_output="softmax"
            )
            self.fold_pred.append(y_pred_i)
            self.fold_true.append(self.y_val_fold[i])

        fold_true_flat = FlattenList(self.fold_true)
        fold_pred_flat = FlattenList(self.fold_pred)
        fold_true_flat = np.array(fold_true_flat).astype("int")
        fold_pred_flat = np.array(fold_pred_flat).astype("int")
        #F1 score
        macro_F1 = macro_f1_metric_keras(fold_true_flat, fold_pred_flat)
        self.macro_F1_scores.append(macro_F1)
        #add mcc_metric_keras
        mcc = mcc_metric_keras(fold_true_flat, fold_pred_flat)
        self.mcc_scores.append(mcc)
        #balanced accuracy
        accuracy = keras.accuracy(fold_true_flat, fold_pred_flat)

        print(f"Epoch {epoch + 1}: Macro F1-Score = {macro_F1:.4f}", f"MCC = {mcc:.4f}", f"Accuracy = {accuracy:.4f}")


#use this callback to evaluate the network using chuncks in validation. For old keras versions without built-in F1
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.macro_F1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = (self.model.predict(X_val) > 0.5).astype(int)
        macro_f1 = metrics.f1_score(y_val[...,0], y_pred[...,0], average='macro')
        self.macro_F1_scores.append(macro_f1)
        print(f"Epoch {epoch+1}: Macro F1 score: {macro_f1:.4f}")        




#tentativi 19/01 per noise addiction WIP
#add noise directly to the datat
def AddLinearInterp(X, n_segments=1, len_segments=120):
    """
    DESCRIPTION
    This function is used to add noise to the input data.
    
    INPUT
    - X: the input data (NumPy array)
    - n_segments: the number of linear segments to add
    - len_segments: the maximum length of the segments to add
    
    OUTPUT
    - X_noisy: the noisy data
    """
    X_noisy = X.copy()  # Work on a copy of the data
    for i in range(n_segments):
        start = np.random.randint(0, X_noisy.shape[0] - len_segments)
        stop = start + np.random.randint(1, len_segments)
        X_noisy[start:stop+1] = np.linspace(X_noisy[start], X_noisy[stop], stop - start + 1)
    return X_noisy

class AddNoiseAtEpochStart(tf.keras.callbacks.Callback):
    def __init__(self, dataset, add_linear_interp_func):
        super().__init__()
        self.dataset = dataset
        self.add_linear_interp_func = add_linear_interp_func

    def on_epoch_begin(self, epoch, logs=None):
        n_segments = np.random.randint(0, 2)  # Randomize for each epoch
        len_segments = int(np.random.uniform(10, 360))  # Randomize for each epoch

        def add_noise(x, y):
            # Convert tensor to NumPy, apply noise, and convert back to tensor
            x_noisy = tf.numpy_function(
                func=self.add_linear_interp_func,
                inp=[x, n_segments, len_segments],
                Tout=tf.float32
            )
            return x_noisy, y

        # Apply noise to the dataset
        self.model.training_data = self.dataset.map(add_noise)



# Define the noise-adding function
def AddLinearInterp(X, n_segments, len_segments):
    """
    Adds linear interpolation noise to the data.
    """
    X_noisy = X.copy()  # Work on a copy to avoid modifying the original data
    for i in range(n_segments):
        start = np.random.randint(0, X_noisy.shape[0] - len_segments)
        stop = start + np.random.randint(1, len_segments)
        X_noisy[start:stop + 1] = np.linspace(X_noisy[start], X_noisy[stop], stop - start + 1)
    return X_noisy

# Define the data generator class
class NoisyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, add_noise_func=AddLinearInterp, 
                 n_segments=1, len_segments=120, shuffle=True, addNoise=True, max_epochs=50):
        """
        Initializes the generator.
        
        Parameters:
        - max_epochs: Total number of epochs to train (used for noise scaling).
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.add_noise_func = add_noise_func
        self.n_segments = n_segments
        self.len_segments = len_segments
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.addNoise = addNoise
        self.current_epoch = 0  # Track current epoch
        self.max_epochs = max_epochs  # Maximum number of epochs

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        if self.addNoise:
            X_batch_noisy = np.array([self.add_noise_func(x, self.n_segments, self.len_segments) for x in X_batch])
            return X_batch_noisy, y_batch
        else:
            return X_batch, y_batch  

    def on_epoch_end(self):
        """Shuffles data and increases noise level at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Increase noise progressively
        self.current_epoch += 1
        progress = self.current_epoch / self.max_epochs  # Normalize from 0 to 1

        # Increase n_segments and len_segments over epochs
        self.n_segments = int(0 + progress * 3)  # Increase up to 3 extra segments
        self.len_segments = int(10 + progress * 350)  # Increase segment length up to 360



# Retrieve the results at the end of training
def retrieve_results(iterations_results, metric, n_splits, n_iterations):
    results = list()
    for n in range(n_iterations):
        results_iterations = list()
        for k in range(n_splits):
            results_iterations.append(iterations_results[n]["fold_"+str(k)][metric])
        results.append(results_iterations)
    results = np.array(results) # [n_iterations, n_folds, n_epochs+1 perchè c'è il pretrain]

    return results

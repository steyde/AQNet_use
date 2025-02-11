import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, accuracy_score, balanced_accuracy_score


def drop_nan_values(x_list):
    """
    DESCRIPTION:
        This function drops NaN values from a list
    INPUT:
        - x_list: the shape of the NaN array
    OUTPUT:
        - x_list_dropped: the list in which "nan" entries have been removed
    """
    x_list_dropped = [x for x in x_list if str(x) != "nan"]
    return x_list_dropped


def create_nan_array(shape):
    """
    DESCRIPTION:
        This function creates an empty NaN array of a predefined shape
    INPUT:
        - shape: the shape of the NaN array
    OUTPUT:
        - nan_array: the created NaN array
    """
    nan_array = np.full(shape, np.nan)
    return nan_array


def fill_nan_values_forward(arr):
    """
    DESCRIPTION:
        This function fills forward the NaN values
    INPUT:
        - arr: the input array in which to fill forward the NaN values
    OUTPUT:
        - out: the ouput interpolated array
    """
    df = pd.DataFrame(arr)
    df.fillna(method="ffill", inplace=True)

    #if the first element is nan, fill backward
    temp = np.isnan(df.iloc[0])
    temp = temp[0]
    if temp:
        df.fillna(method="bfill", inplace=True)
        
    out = df.values.flatten()

    return out

#same as fill_nan_values_forward but only if the nan segments last less than 120 samples, else substitute with 0.5
def fill_nan_values_forward_short_nan(arr):
    """
    DESCRIPTION:
        This function fills forward the NaN values, but if the nan segments last more than 120 samples, substitute with 0.5
    INPUT:
        - arr: the input array in which to fill forward the NaN values
    OUTPUT:
        - out: the ouput interpolated array
    """
    df      = pd.DataFrame(arr)
    input_s = df.copy()
    
    df.fillna(method="ffill", inplace=True)

    #if the first element is nan, fill backward
    temp = np.isnan(df.iloc[0])
    temp = temp[0]
    if temp:
        df.fillna(method="bfill", inplace=True)
        
    out     = df.values.flatten()
    input_s = input_s.values.flatten()

    #substitute with 0.5 if nan segments last more than 120 samples
    for i in range(1,len(out)):
        if np.isnan(input_s[i]) and not np.isnan(input_s[i-1]):
            start = i
        if not np.isnan(input_s[i]) and np.isnan(input_s[i-1]):
            stop = i
            if stop-start>120:
                out[start:stop] = 0.5

    return out


def compute_mcc(y_true, y_pred):
    """
    DESCRIPTION:
        This compute the Matthews Correlation Coefficient (MCC) between the true and predicted annotations array
    INPUT:
        - y_true: the binary true annotation array from the clinician
        - y_pred: the binary predicted annotation array
    OUTPUT:
        - mcc_score: the computed MCC score
    """
    tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    tn = np.sum(np.round(np.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = np.sum(np.round(np.clip((1 - y_true) * y_pred, 0, 1)))
    fn = np.sum(np.round(np.clip(y_true * (1 - y_pred), 0, 1)))

    # Scaling to avoid overflow
    scale = np.max([tp, tn, fp, fn])
    if scale>0.01:
        tp = tp / scale
        tn = tn / scale
        fp = fp / scale
        fn = fn / scale
    else:
        scale = 1

    # Numerator and denominator of MCC
    num = tp * tn - fp * fn
    den = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn + fn)
    # Below we have the various cases (note: they have to be kept in this order!)
    # 1. Standard case
    if num != 0 and den != 0:
        mcc_score = num / den
    # 2. Case in which we have 0/0 --> indeterminate form
    if num == 0 and den == 0:
        mcc_score = 0
    # 3. Case in which they agree 100%
    if tp*scale == len(y_true):
        mcc_score = 1
    # 4. Case in which they do not agree at all
    if fp*scale == len(y_true):
        mcc_score = -1
    return mcc_score


def create_annotation_list(start_minute, stop_minute, state_list, time, fs):
    """
    DESCRIPTION:
        This creates an annotation list from the read excel/csv file
    INPUT:
        - start_minute:
        - stop_minute:
        - state_list:
        - time:
        - fs:
    OUTPUT:
        - annotation_list: the computed MCC score
    """
    # Creating a dictionary to map the states to integers
    state_map = {"Q": 0, "1": 0, "1.0": 0, "2": 1,"2.0": 1,"3": 0,"3.0": 0,"4": 1,"4.0": 1, "5" : -1, "A": 1, "IND": -1, "i": -1}
    # Initializing the array to NaN values
    annotation_list = create_nan_array(time.shape)
    # Creating the annotation list
    for i in range(len(state_list)):
        start_samples = int(start_minute[i]*60*fs)
        stop_samples = int(stop_minute[i]*60*fs)
        annotation_list[start_samples:stop_samples] = state_map[str(state_list[i])]
    # Imposing NaN values to the last available state which is not NaN (i.e., filling NaNs)
    annotation_list = fill_nan_values_forward(annotation_list)
    # Setting indeterminate states to NaN values
    annotation_list[annotation_list == -1] = np.nan
    return annotation_list


def FindStartStopStates(arr):
    """
    DESCRIPTION:
        This function find the start and stop values of a state array
    INPUT:
        - arr: the array in which to find the start and stop values (indexes) of the array values
    OUTPUT:
        - state_list: the list of states
        - start_stop_list: the list of start, stop values for each state
    """

    state_list = list()
    start_stop_list = list()
    # Initializing at the first element
    state_list.append(arr[0])
    start = 0
    start_stop_list.append(start)
    # Appending only when it changes
    for i in range(len(arr)-1):
        if arr[i] != arr[i+1]:
            stop = i
            start_stop_list.append(stop)
            state_list.append(arr[i+1])
            start = stop+1
            start_stop_list.append(start)
    # Appending the last change
    start_stop_list.append(len(arr)-1)
    start_stop_list = np.array(start_stop_list).reshape(-1, 2)
    return state_list, start_stop_list


def format_annotations(annotations, file_names, time, fs):
    """
    DESCRIPTION:
        This function reads the read CSV file and extracts the annotation array using the create_annotation_list() function
    INPUT:
        - annotations: the read CSV file
        - file_names: the name of the file in order to match the signas and annotations
        - time: the time array
        - fs: the sampling frequency
    OUTPUT:
        - annotations_list: the annotations returned as a list
    """

    # Cycling on the annotations and getting the correct annotation for the file names
    annotations_list = list()
    for i in range(len(annotations)):
        # Finding the correct annotations
        try:
            mask = annotations["Final File Name"] == file_names[i]
            selected_annot = annotations.loc[mask] 
        except:
            mask = annotations["Study ID"] == file_names[i]
            selected_annot = annotations.loc[mask] 
        # Retrieving the list of [start, stop, state, ... and so on]
        selected_annot = selected_annot.values.tolist()[0][2:-4]
        # Extracting the elements of each start, stop, state
        start_minute = drop_nan_values(selected_annot[0::3])
        stop_minute = drop_nan_values(selected_annot[1::3])
        state_list = drop_nan_values(selected_annot[2::3])

        # Creating the list of annotations so that we have the correct state at the correct time point of the time array
        annotations_list.append(create_annotation_list(start_minute, stop_minute, state_list, time[i], fs))
    return annotations_list


def format_annotations_napami_to_columbia(annotations, file_names, time, fs, partial_annotations = False):
    """
    DESCRIPTION:
        This function reads the read CSV file and extracts the annotation array using the create_annotation_list() function
        This function is for the data that we sent to columbia to annotate from the Napami database
        (Da sistemare)
    INPUT:
        - annotations: the read CSV file
        - file_names: the name of the file in order to match the signas and annotations
        - time: the time array
        - fs: the sampling frequency
        - partial_annotations: the IDs of the missing annotations
    OUTPUT:
        - annotations_list: the annotations returned as a list
    """

    if partial_annotations:
        #store the ones to check latyer
        missing_annotations = []

    # Cycling on the annotations and getting the correct annotation for the file names
    annotations_list = list()
    for i in range(len(file_names)):
        # Finding the correct annotations
        mask = annotations["CTG_recordings ID"] == file_names[i]
        if np.sum(mask) == 0:
            if partial_annotations:
                missing_annotations.append(file_names[i])
            else:
                print(f"File '{file_names[i]}' not found in the annotations")
                start_minute = [0]
                stop_minute  =  [np.round(time[i][-1]/60)]
                state_list = ["IND"]
        else:
            selected_annot = annotations.loc[mask] 
            # Retrieving the list of [start, stop, state, ... and so on]
            selected_annot = selected_annot.values.tolist()[0][2:-3]

            #check that the annotation is not empty
            if np.isnan(selected_annot[0]):
                print(f"File '{file_names[i]}' has empty annotations")
                start_minute = [0]
                stop_minute  =  [np.round(time[i][-1]/60)]
                state_list = ["IND"]
            else:
                # Extracting the elements of each start, stop, state
                start_minute = drop_nan_values(selected_annot[0::3])
                stop_minute = drop_nan_values(selected_annot[1::3])
                state_list = drop_nan_values(selected_annot[2::3])

        # Creating the list of annotations so that we have the correct state at the correct time point of the time array
        
        if file_names[i]==22164:
            #for some reasons the annptations are shifted...
            warnings.warn("Shifting annotations for file 22164")
            #AnnotList = AnnotList[int(np.round(60*fs*18)):]
            start_minute = [max(0,x - 18) for x in start_minute]
            stop_minute  = [max(0,x - 18) for x in stop_minute]
            
        if file_names[i]==9634:
            warnings.warn("Shifting annotations for file 9634")
            #AnnotList = AnnotList[int(np.round(60*fs*7.5)):]
            start_minute = [max(0,x - 7.5) for x in start_minute]
            stop_minute  = [max(0,x - 7.5) for x in stop_minute]

        if file_names[i]==865:
            warnings.warn("Shifting annotations for file 9634")
            #AnnotList = AnnotList[int(np.round(60*fs*7.5)):]
            start_minute = [max(0,x - 7) for x in start_minute]
            stop_minute  = [max(0,x - 7) for x in stop_minute]
        
        AnnotList = create_annotation_list(start_minute, stop_minute, state_list, time[i], fs)
            
        annotations_list.append(AnnotList)

    if partial_annotations:
        return annotations_list, missing_annotations
    else:
    
        return annotations_list
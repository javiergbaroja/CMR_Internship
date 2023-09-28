
import os

import pandas as pd
import numpy as np
import numpy.ma as ma
import random
import torch

from skimage.morphology import binary_opening, disk
from skimage.morphology import binary_dilation, disk
from sklearn.model_selection import StratifiedKFold, train_test_split

from Helpers.utils import parse_json_file, save_json

def adjust_order(positions:np.ndarray, imgs:np.ndarray):

    imgs = [x for _,x in sorted(zip(positions.tolist(), imgs.tolist()))]

    return np.array(imgs)

def fix_sequence_order(positions:str, images:str, labels:str, fix_order:bool) -> tuple:
    
    positions = np.load(positions)
    images = np.load(images)
    labels = np.load(labels)

    if not all(positions == sorted(positions)) and fix_order:
        images = adjust_order(positions, images)
        labels = adjust_order(positions, labels)

    return images, labels

def load_masks_and_img_from_path(path:str, fix_order:bool):
    positions = os.path.join(path, 'positions.npy')
    image_file_to_load = os.path.join(path, 'images.npy')
    label_file_to_load = os.path.join(path, 'labels.npy')

    return fix_sequence_order(positions, image_file_to_load, label_file_to_load, fix_order)


def load_all_data(cohorts:list, path:str, method:str, fix_order:bool) -> tuple:
    images, labels, pids = {}, {}, {}
    for cohort in cohorts:
        images[cohort] = {}
        labels[cohort] = {}
        pids[cohort] = []
        
        cohort_data_folder = os.path.join(path, cohort)
        for pdir in os.listdir(cohort_data_folder):
            if pdir[0] == 'p':
                loaded_image, loaded_label = load_masks_and_img_from_path(os.path.join(cohort_data_folder, pdir), fix_order)
                images[cohort][pdir] = loaded_image
                labels[cohort][pdir] = get_label(loaded_label, method)
                pids[cohort].append(pdir)
    
    return images, labels, pids


def load_partitioned_data(fixed_sets:str, method:str, fix_order:bool) -> tuple:

    partitions = parse_json_file(fixed_sets)

    output = tuple()

    for partition in partitions.values():

        data = {"images": [],
                "masks": [],
                "pIDs": [],}
        
        for pdir in partition:

            loaded_image, loaded_label = load_masks_and_img_from_path(pdir, fix_order)
            loaded_label = get_label(loaded_label, method)
            pid = "_".join([os.path.split(os.path.split(pdir)[0])[1], os.path.split(pdir)[1]])

            data["images"].append(loaded_image.astype('float32')[..., None])
            data["masks"].append(loaded_label.astype('float32'))
            data["pIDs"].append(pid)
        output += (data, )

    return output

def denoise_slice(input_slice:np.ndarray, method:str):

    # Process sd: 1) sd2 + sd5, 2) opening 3) dilation

    tmp_after = np.copy(input_slice)
    a = np.squeeze(tmp_after[:,:,1] + tmp_after[:,:,2]) if method =="SD" else tmp_after[:,:,1] # 1) sd2 + sd5 | fwhm
    msk = ma.masked_where(a > 0, a, copy=True)
    b = binary_opening(msk, disk(1.0)) # 2) Opening
    c = binary_dilation(b,disk(1.0)) # 3) Dilation

    tmp_after = np.copy(input_slice)
    cc = ma.masked_where(c> 0, c, copy=True)
    cc[cc>0] = 1
    if method == "SD":
        tmp_after_red = np.multiply(tmp_after[:,:,1],cc) # Apply corrected scar to SD2
        tmp_after_red[tmp_after_red>0] = 1
        tmp_after_blue = np.multiply(tmp_after[:,:,2],cc) # Apply corrected scar to SD5
        tmp_after_blue[tmp_after_blue>0] = 1

        tmp_after = np.copy(input_slice)
        tmp_after_green = tmp_after[:,:,0] #np.sum(tmp_after[...], axis = 2)
        tmp = np.zeros_like(tmp_after)
        #tmp_after_green = binary_closing(tmp_after_green, disk(1.5))
        tmp_after_green[tmp_after_red == 1] = 0
        tmp_after_green[tmp_after_blue == 1] = 0
        
        tmp[:,:,0] = tmp_after_green # Assign Myocardium
        tmp[:,:,1] = tmp_after_red # Assign SD2
        tmp[:,:,2] = tmp_after_blue # Assign SD5
    else:
        tmp_after_blue = np.multiply(tmp_after[:,:,1], cc) # Apply corrected scar to fwhm
        tmp_after_blue[tmp_after_blue>0] = 1

        tmp_after = np.copy(input_slice)
        tmp_after_green = tmp_after[:,:,0] #np.sum(tmp_after[...], axis = 2)
        tmp= np.zeros_like(tmp_after)
        #tmp_after_green = binary_closing(tmp_after_green, disk(1.5))
        tmp_after_green[tmp_after_blue == 1] = 0
        
        tmp[:,:,0] = tmp_after_green # Assign Myocardium
        tmp[:,:,1] = tmp_after_blue # Assign FWHM

    return tmp


def denoise_msk(input_data: np.ndarray, method:str="SD") -> np.ndarray:
    """This function remove residual noise from the masks by morphological operations.

    Args:
        input_data (np.ndarray): label array of shape [#nslices, H, W, C]
        method (str, optional): Label method chosen for training. Defaults to 'SD'. 
            If 'SD': input_data.shape[-1] = 3
            If 'FWHM': input_data.shape[-1] = 2

    Returns:
        np.ndarray: Denoised mask array of shape [#nslices, H, W]
    """

    denoised =  np.zeros(input_data.shape)
    for i in range(input_data.shape[0]):
        denoised[i,...] = denoise_slice(input_data[i,...], method)
    return denoised


def get_label(arr:np.ndarray, method:str='SD') -> np.ndarray:    
    """This function takes the label array and,
      depending on the type of prediction chosen for training, filters them to only contain myocardium and scar information.
      The channels as stored are:
      0:Epicardium
      1:Endocardium/Lument
      2:myocardium minus healthy patches
      3,4:healthy patches 1,2
      6:myocardium (all)
      7:sd2
      8:sd5
      9:fwhm

    Args:
        arr (np.ndarray): label array, unfiltered
        method (str, optional): Label method chosen for training. Defaults to 'SD'.

    Returns:
        np.ndarray: Label array, filtered
    """

    if method == 'SD':
        indices = [6, 7, 8]
        labels = np.sum(denoise_msk(arr[...,indices], method), axis=3)
    elif method == 'FWHM':
        indices = [6, 9]
        labels = np.sum(denoise_msk(arr[...,indices], method), axis=3)
    else:
        raise ValueError("'method' must be chosen from SD or FWHM")

    return labels

def select_all(complete_list:list, array, unused=None):
    complete_list.append(torch.from_numpy(array))
    return complete_list

def select_central(complete_list:list, array, n_slices:int):
    array = array[(len(array)//2 - n_slices//2) : (len(array)//2 + n_slices//2)]
    complete_list.append(torch.from_numpy(array))
    return complete_list

def update_pids_central_mode(pid_list, pid, unusused1=None, unusused2=None):
    pid_list.append(pid)
    return pid_list

def select_windows(complete_list:list, array:np.ndarray, n_slices:int):
    array_list = []
    for i in range(len(array)-n_slices+1):
        array_list.append(torch.from_numpy(array[i:i+n_slices]))

    complete_list.extend(array_list)
    return complete_list

def update_pids_window_mode(pid_list, pid, len_array, n_slices):
    pids = [pid for __ in range(len_array-n_slices+1)]
    pid_list.extend(pids)
    return pid_list

def select_method(n_slices_central, n_slices_window):
    if n_slices_central != -1 and n_slices_window == -1:
        n_slices = n_slices_central
        select_slices = select_central
        append_pids = update_pids_central_mode
    elif n_slices_central == -1 and n_slices_window != -1:
        n_slices = n_slices_window
        select_slices = select_windows
        append_pids = update_pids_window_mode
    else:
        n_slices = -1
        select_slices = select_all
        append_pids = update_pids_central_mode

    return select_slices, append_pids, n_slices

def preprocessing_2D(data:dict) -> dict:
    """Image Preprocessing to be used when the segmentation model runs only at the slice level.

    Args:
        data (dict)
            images (list): All the images from all the patients. Each item.shape = (#slices, H, W, C)
            masks (list): All the label masks from all the patients. Each item.shape = (#slices, H, W)
            pIDs (list): List with the patient identifiers. As many items as patients
    
    Returns:
        dict: 
            images (torch.Tensor): image data of shape (#images, C, H, W)
            masks (torch.Tensor): label data of shape (#images, C, H, W)
            pIDs (list): List with the patient identifiers.

    """

    X, Y = np.concatenate(data["images"], axis=0), np.concatenate(data["masks"], axis=0) # (#images, H, W, C), (#images, H, W, C)

    for i, ID in enumerate(data["pIDs"]):
        if i == 0: pIDs = [ID for __ in range(data["images"][i].shape[0])]
        else: pIDs.extend([ID for __ in range(data["images"][i].shape[0])])

    # center crop
    X = X[:, 8:-8, 8:-8]
    Y = Y[:, 8:-8, 8:-8]

    to_keep = np.sum(Y, (1,2)) != 0
    X = X[to_keep]
    Y = Y[to_keep]
    pIDs = np.array(pIDs)[to_keep]

    return {"images": torch.from_numpy(np.transpose(X, (0,3,1,2))),
            "masks": torch.from_numpy(np.expand_dims(Y, axis=1)),
            "pIDs": pIDs}


def preprocessing_3D(data:dict, n_slices_central:int=-1, n_slices_window:int=-1) -> tuple:
    """Preprocessing for the case where the patient data should not be mixed.

    Args:
        data (dict)
            images (list): All the images from all the patients. Each item.shape = (#slices, H, W, C)
            masks (list): All the label masks from all the patients. Each item.shape = (#slices, H, W)
            pIDs (list): List with the patient identifiers. As many items as patients
        n_slices_central (int):

    Returns:
        tuple: 
            list: one item per patient (images). Each item is a torch.Tensor of item.shape (#slices, H, W, C)
            list: one item per patient (masks). Each item is a torch.Tensor of item.shape (#slices, H, W, C)
            if n_slices_central != -1: all items in the list have the same #slices
    """

    append_slices, append_pids, n_slices = select_method(n_slices_central, n_slices_window)

    assert len(data["images"]) == len(data["masks"])
    X_new, Y_new, pIDS_new = [], [], []
    
    for i in range(len(data["images"])):

        X_aux = data["images"][i][:, 8:-8, 8:-8] # Trim edges of slices
        Y_aux = data["masks"][i][:, 8:-8, 8:-8]

        to_keep = np.sum(Y_aux, (1,2)) != 0
        X_aux = X_aux[to_keep]
        Y_aux = Y_aux[to_keep]

        if len(X_aux) < n_slices: 
                continue
        
        X_new =  append_slices(X_new, np.transpose(X_aux, (0,3,1,2)), n_slices)
        Y_new =  append_slices(Y_new, np.expand_dims(Y_aux, axis=1),  n_slices)
        pIDS_new = append_pids(pIDS_new, data["pIDs"][i], len(X_aux), n_slices)

    return {"images": X_new,
            "masks": Y_new,
            "pIDs": pIDS_new}
                        

def preprocessing(data:dict, views:str, n_slices_central:int=-1, n_slices_window:int=-1) -> tuple:
    """Receives X and Y that contain per patient image data and label masks respectively. 
            If views == "2D" there will be no per patient distinction.

    Args:
        data (dict)
            images (list): All the images from all the patients. Each item.shape = (#slices, H, W, C)
            masks (list): All the label masks from all the patients. Each item.shape = (#slices, H, W)
            pIDs (list): patient IDs
        views (str): Determines whether the images will be preprocessed as 3D or 2D
        n_slices_central (int, opt): In case of sequential consideration, Nr. of central slices to select per patient, with one volume per patient.
        n_slices_window (int, opt): In case of sequential consideration, Nr. of slices in each sample. Several volumes extracted per patient in an overlapping window approach.
    Returns:
        tuple: contains the image data and labels after preprocessing. 
            See preprocessing_2D and preprocessing_3D for more information.
    """

    if views =="2D":
       return preprocessing_2D(data)

    elif views == "3D":
        return preprocessing_3D(data, n_slices_central, n_slices_window)
    else:
        raise ValueError("views should be a string containing either '2D' or '3D'.")
    

def train_val_test_split(images:dict, labels:dict, pids:dict, shuffle:bool=False):
    """
    Split the patients of each cohort into 70% train, 15% validation and 15% test
    """
    
    train_data = {"pIDs": [], "images": [], "masks": []}
    valid_data = {"pIDs": [], "images": [], "masks": []}
    test_data = {"pIDs": [], "images": [], "masks": []}
    # get number of patients in each split for each cohort
    for cohort in images.keys():
        n_patients = len(images[cohort].keys())
        n_test = int(np.ceil(0.15 * n_patients))
        if n_patients - n_test > 0:
            n_val = int(np.ceil(0.15 * n_patients))
        else: 
            n_val = 0
        if n_patients - n_test - n_val > 0:
            n_train = n_patients - n_test - n_val
        else:
            n_train = 0
        
        # random split the patients into train, val and test
        patients = list(images[cohort].keys())
        if shuffle:
            random.shuffle(patients)
        train_patients = patients[:n_train]
        val_patients = patients[n_train:(n_train+n_val)]
        test_patients = patients[(n_train+n_val):(len(patients)+1)]

        for p in train_patients:
            train_data["images"].append(images[cohort][p].astype('float32')[...,None])
            train_data["masks"].append(labels[cohort][p].astype('float32'))
            train_data["pIDs"].append("_".join([cohort, p]))
        
        for p in val_patients:
            valid_data["images"].append(images[cohort][p].astype('float32')[...,None] )
            valid_data["masks"].append(labels[cohort][p].astype('float32'))
            valid_data["pIDs"].append("_".join([cohort, p]))
        
        for p in test_patients:
            test_data["images"].append(images[cohort][p].astype('float32')[...,None] )
            test_data["masks"].append(labels[cohort][p].astype('float32'))
            test_data["pIDs"].append("_".join([cohort, p]))
 
    return train_data, valid_data, test_data

def create_cv_splits(config, logger):

    splits_path = "tmp"
    config.fixed_sets = splits_path
    os.makedirs(splits_path, exist_ok=True)

    metadata = pd.read_csv(os.path.join(config.data_folder, "data_summary.csv"))
    metadata = metadata[metadata['useful_n_slices_SD']>=8.].reset_index(drop=True)

    # Division Train/Test -> 85/15%
    train, test = train_test_split(metadata, test_size=42, shuffle=True, random_state=config.seed, stratify=metadata.Hospital)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    patient_info = train[["Hospital","PatientID"]]
    patient_info = [os.path.join(patient_info.Hospital[i], patient_info.PatientID[i]) for i in range(len(patient_info))]

    train_cv = []
    val_cv = []
    skf = StratifiedKFold(n_splits=config.n_cross_val, random_state=config.seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(patient_info, list(train.Hospital))):
        train_cv.append(sorted([patient_info[i] for i in train_index]))
        val_cv.append(sorted([patient_info[i] for i in test_index]))
        logger.info(f"Fold {i}:")
        logger.info(f"  Training  : {len(train_index)}")
        logger.info(f"  Validation:  {len(test_index)}")
    
    test_set = [os.path.abspath(os.path.join(config.data_folder, test.loc[i, "Hospital"], test.loc[i, "PatientID"])) for i in range(len(test))]
    
    for i in range(config.n_cross_val):
        train_set = [os.path.abspath(os.path.join(config.data_folder, train_cv[i][j]) )for j in range(len(train_cv[i]))]
        val_set = [os.path.abspath(os.path.join(config.data_folder, val_cv[i][j])) for j in range(len(val_cv[i]))]

        partition = {"train": sorted(train_set),
                     "val": sorted(val_set),
                     "test": sorted(test_set)}
        
        save_json(splits_path, file_name=f"data_cv_{i}.json", object=partition)

    
    return config, splits_path
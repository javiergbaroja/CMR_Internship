import re
import os
import numpy as np



def backslash2slash(path:str) -> str:
    return re.sub(r'\\',r'/',path)


def retrieve_patient_info(patient_folder:str, relative:int=None) -> dict:
    """Returns a dictionary containing the information of a patient that comprises one row in the data summaries csv. 

    Args:
        patient_folder (str): path to folder containing the data of one patient.
        relative (int): number indicating the label [0-9] that the items in the output dict should be relative to.
            Example: relative=6 --> myocardium label --> rest given as #pixels/#myo pixels
            Defaults to None --> Relative to the total number of voxels in the volume.

    Returns:
        dict: extracted information from patient data:
            n_voxels (int): total number of voxels in the patient volume.
            Hospital (str): hospital that provided the images.
            PatientID (str): patient number. Hospital dependent, not UUID.
            n_slices (int): number of slices in the volume.
            ordered_slices (bool): Whether or not the slices are loaded in the correct sequential order from images.npy file.
            useful_n_slices_SD (int): number of slices in the volume with SD label.
            useful_n_slices_FWHM (int): number of slices in the volume with FWHM label.
            slice_dim (np.array): dimensions of a 2D slice in axial direction.
            Quality (dict): Result of QC classifier. Result summary.
            Slice_thickness (np.array)
            Epicardium (int)
            Endocardium|Lumen (int)
            Myocardium-healthy_patches (int)
            Healthy_patch_1 (int)
            Healthy_patch_2 (int)
            Empty (int)
            Myocardium (int)
            Scar_SD2 (int)
            Scar_SD5 (int)
            Scar_FWHM (int)
    """
    
    # Load data
    labels = np.load(patient_folder + "/labels.npy")
    # image = np.load(os.path.join(patient_folder, "images.npy"))
    positions = np.load(os.path.join(patient_folder, "positions.npy"))
    quality = np.load(os.path.join(patient_folder, "qualities.npy"))
    thickness = np.load(os.path.join(patient_folder, "slice_thickness.npy"))
    
    count_dict = {}
    label_names = [
        "Epicardium", 
        "Endocardium|Lumen",
        "Myocardium-healthy_patches",
        "Healthy_patch 1",
        "Healthy_patch 2",
        "Empty",
        "Myocardium",
        "Scar_SD2",
        "Scar_SD5",
        "Scar_FWHM"]
    
    # Fill in dictionary 
    count_dict["n_slices"] = labels.shape[0]
    indices = [6,9]
    take = np.sum(np.sum(labels[...,indices], axis=3), (1,2))!= 0
    count_dict["useful_n_slices_FWHM"] = sum(take)
    indices = [6,7,8]
    take = np.sum(np.sum(labels[...,indices], axis=3), (1,2))!= 0
    count_dict["useful_n_slices_SD"] = sum(take)
    

    labels = np.round(labels[take,...])
    count_dict["Dice_SD2_FWHM"] = dice_score(array1=labels[...,7], array2=labels[...,9])
    count_dict["Dice_SD5_FWHM"] = dice_score(array1=labels[...,8], array2=labels[...,9])
    count_dict["ordered_slices"] = all(positions == sorted(positions))
    count_dict["n_voxels"] = len(labels[...,0].flatten())
    count_dict["Hospital"] = os.path.basename(os.path.dirname(patient_folder))
    count_dict["PatientID"] = os.path.basename(patient_folder)
    count_dict["slice_dim"] = labels.shape[1:-1]
    count_dict["Quality"] = dict(np.array(np.unique(quality, return_counts=True)).T)
    count_dict["Slice_thickness"] = thickness
    count_dict["Scar_SD2"] = np.sum(np.sum(labels[...,[7,8]], axis=3)>=1.)/np.sum(labels[...,relative])
    count_dict["Scar_SD5"] = np.sum(labels[...,8])/np.sum(labels[...,relative])
    count_dict["Scar_FWHM"] = np.sum(labels[...,9])/np.sum(labels[...,relative])


    # Amounts relative to the total number of voxels.
    assert len(label_names) == labels.shape[-1]
    for i in range(labels.shape[-1]-3):
        if relative is not None:
            count_dict[label_names[i]] = np.sum(labels[...,i]) / np.sum(labels[...,relative])
        else:
            count_dict[label_names[i]] = np.sum(labels[...,i]) / count_dict["n_voxels"]
    


    return count_dict

def str2dict(string:str) -> dict:
    string = string.split("{")[1].split("}")[0]
    items = string.split(", ")
    dictionary = {}
    for item in items:
        dictionary[eval(item.split(": ")[0])] = eval(eval(item.split(": ")[1]))
    return dictionary

def dice_score(array1:np.array, array2:np.array):
    if array1.shape != array2.shape:
        raise ValueError("Shape mismatch: array1 and array2 must have the same shape.")
    
    array1 = array1.astype(bool)
    array2 = array2.astype(bool)

    sum_array = array1.sum() + array2.sum()
    intersect_array = np.logical_and(array1, array2).sum()

    return 2*intersect_array / sum_array
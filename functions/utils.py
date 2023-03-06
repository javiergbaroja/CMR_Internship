import re
import os
import numpy as np


def backslash2slash(path:str) -> str:
    return re.sub(r'\\',r'/',path)


def retrieve_patient_info(patient_folder:str) -> dict:
    """Returns a dictionary containing the information of a patient that comprises one row in the data summaries csv. 

    Args:
        patient_folder (str): path to folder containing the data of one patient.

    Returns:
        dict: extracted information from patient data:
            n_voxels (int): total number of voxels in the patient volume.
            Hospital (str): hospital that provided the images.
            PatientID (str): patient number. Hospital dependent, not UUID.
            n_slices (int): number of slices in the volume.
            slice_dim (np.array): dimensions of a 2D slice in axial direction.
            Quality (dict): Result of QC classifier. Result summary.
            Slice_thickness (np.array)
            Epicardium (int)
            Endocardium|Lumen (int)
            Myocardium - healthy patches (int)
            Healthy patch 1 (int)
            Healthy patch 2 (int)
            Empty (int)
            Myocardium (int)
            Scar SD2 (int)
            Scar SD5 (int)
            Scar FWHM (int)
    """
    
    # Load data
    labels = np.load(patient_folder + "/labels.npy")
    # image = np.load(os.path.join(patient_folder, "images.npy"))
    # positions = np.load(os.path.join(patient_folder, "positions.npy"))
    quality = np.load(os.path.join(patient_folder, "qualities.npy"))
    thickness = np.load(os.path.join(patient_folder, "slice_thickness.npy"))
    
    count_dict = {}
    label_names = [
        "Epicardium", 
        "Endocardium|Lumen",
        "Myocardium - healthy patches",
        "Healthy patch 1",
        "Healthy patch 2",
        "Empty",
        "Myocardium",
        "Scar SD2",
        "Scar SD5",
        "Scar FWHM"]
    
    # Fill in dictionary 
    count_dict["n_voxels"] = len(labels[...,0].flatten())
    count_dict["Hospital"] = os.path.basename(os.path.dirname(patient_folder))
    count_dict["PatientID"] = os.path.basename(patient_folder)
    count_dict["n_slices"] = labels.shape[0]
    count_dict["slice_dim"] = labels.shape[1:-1]
    count_dict["Quality"] = dict(np.array(np.unique(quality, return_counts=True)).T)
    count_dict["Slice_thickness"] = thickness

    # Amounts relative to the total number of voxels.
    assert len(label_names) == labels.shape[-1]
    for i in range(labels.shape[-1]):
        count_dict[label_names[i]] = np.sum(labels[...,i]) / count_dict["n_voxels"]

    return count_dict
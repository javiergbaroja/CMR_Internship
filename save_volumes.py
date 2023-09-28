#!/usr/bin/env python3

from Helpers.args import get_args_volume_save as get_args
from Helpers.utils import *

from Trainers.unet_predictor import Predictor

from DataLoader.cohorts import all_cohorts
import pandas as pd
import numpy as np
import vtk
from vtk.util import numpy_support
from torchmetrics import ConfusionMatrix
import skimage
import copy
from argparse import Namespace

def get_scar_ratio(tensor:torch.Tensor):
    scar = ((tensor >= 2.) *1.).sum().item()
    myo = ((tensor >= 1.) *1.).sum().item()
    return scar / myo

def get_scar_ratio_cont(tensor:torch.Tensor):
    vals = []
    for i in range(len(tensor)-1):
        i1 = tensor[i,0,:,:]
        i2 = tensor[i+1,0,:,:]

        scar1 = ((i1>=2.) * 1.).sum().item()
        scar2 = ((i2>=2.) * 1.).sum().item()

        myo1 = ((i1>=1.) * 1.).sum().item()
        myo2 = ((i2>=1.) * 1.).sum().item()

        if myo1 ==0 or myo2 ==0:
            continue

        rat1 = scar1 / myo1
        rat2 = scar2 / myo2

        vals.append(np.abs(rat1 - rat2))
    return np.std(vals)

def get_label_cont_mean_std(tensor:torch.Tensor):
    vals = []
    for i in range(len(tensor)-1):
        i1 = tensor[i,0,:,:]
        i2 = tensor[i+1,0,:,:]

        vals.append(((i1 == i2)*1.).sum().item() / len(i1.flatten()))
    return np.mean(vals), np.std(vals)


def get_label_cont_std(tensor:torch.Tensor):
    return get_label_cont_mean_std(tensor)[1]


def get_label_cont_mean(tensor:torch.Tensor):
    return get_label_cont_mean_std(tensor)[0]



def get_surface2vol_ratio(tensor:torch.Tensor, class_value:int):
    image = copy.deepcopy(tensor[:,0,:,:])
    image[image<class_value] = 0.
    image[image>=class_value] = 1.

    shape = image.shape
    surface = 0
    volume = 0
    for i, slice in enumerate(image):
        volume += slice.sum().item()
        if i == 0 or i == shape[0] - 1:
            surface += slice.sum().item()
        else:
            surface += np.sum(skimage.feature.canny(slice.numpy().astype(np.float32), sigma=3))
    return surface, volume, surface/volume

def get_euler(tensor:torch.Tensor, label:float) -> int:
    volume = copy.deepcopy(tensor.squeeze(1).permute(1,2,0).detach().cpu().numpy())
    volume[volume!=label] = 0.
    volume[volume==label] = 1.
    return skimage.measure.euler_number(volume, connectivity=1)

def get_connected_componets(tensor:torch.Tensor, label):
    volume = copy.deepcopy(tensor.squeeze(1).detach().cpu().numpy())
    volume[volume<label] = 0.
    volume[volume>=label] = 1.
    return skimage.measure.label(volume, connectivity=3, return_num=True)[1]


def get_dice(confusion_matrix, channel):
    TP = torch.sum(confusion_matrix[channel:, channel:])
    FN_FP = torch.sum(confusion_matrix) - torch.sum(confusion_matrix[0:channel,0:channel]) - TP
    dice = (2*TP)/(2*TP + FN_FP)
    return dice


def save_volume(tensor:torch.Tensor, file_name:str, factor=float):

    image = copy.deepcopy(tensor).squeeze(1).detach().cpu().numpy() + factor
    image[image==factor] = 0
    data_type = vtk.VTK_FLOAT
    D, H, W = image.shape

    vtk_data = numpy_support.numpy_to_vtk(num_array=image.ravel(order="F"), deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(D, H, W)
    
    img.SetSpacing([10, 1, 1]) 
    img.SetOrigin([0, 0, 0])

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(img)
    writer.Write()

def save_volumes_and_table(gt:list, preds:list, pids:list, network_name:str, factor:float, folder:str=r"Results\volumes"):

    os.makedirs(folder, exist_ok=True)

    confmat = ConfusionMatrix(task="multiclass", num_classes=4)

    df = pd.DataFrame(columns=["pID", 
                               f"dice_myo", 
                               f"dice_sd2",
                               f"dice_sd5",  

                               f"concomp_myo_pred", 
                               f"concomp_myo_gt",
                               f"concomp_sd2_pred", 
                               f"concomp_sd2_gt",
                               f"concomp_sd5_pred",
                               f"concomp_sd5_gt",

                               f"scar_ratio_pred",
                               f"scar_ratio_gt",

                               f"scar_ratio_cont_pred",
                               f"scar_ratio_cont_gt",

                               f"label_cont_pred",
                               f"label_cont_gt",

                               f"s2vRatio_myo_pred",
                               f"s2vRatio_myo_gt",
                               f"s2vRatio_sd2_pred",
                               f"s2vRatio_sd2_gt",
                               f"s2vRatio_sd5_pred",
                               f"s2vRatio_sd5_gt",
                               f"network_name"])

    for i in range(len(pids)):
        path = os.path.join(folder, pids[i])
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving to : {path}")

        save_volume(tensor=preds[i], file_name=os.path.join(os.getcwd(), path, f"{network_name}_vol.mhd"), factor=factor)
        save_volume(tensor=gt[i], file_name=os.path.join(os.getcwd(), path, "gt_vol.mhd"), factor=0.)

        df.loc[i] = [
            pids[i],
            get_dice(confmat(preds[i][:,0,:,:].flatten(),   gt[i].detach().cpu().flatten()), 1).item(),
            get_dice(confmat(preds[i][:,0,:,:].flatten(),   gt[i].detach().cpu().flatten()), 2).item(),
            get_dice(confmat(preds[i][:,0,:,:].flatten(),   gt[i].detach().cpu().flatten()), 3).item(),

            get_connected_componets(preds[i],   1.),
            get_connected_componets(gt[i].detach().cpu().unsqueeze(1),   1.),
            get_connected_componets(preds[i],   2.),
            get_connected_componets(gt[i].detach().cpu().unsqueeze(1),   2.),
            get_connected_componets(preds[i],   3.),
            get_connected_componets(gt[i].detach().cpu().unsqueeze(1),   3.),

            get_scar_ratio(preds[i]),
            get_scar_ratio(gt[i].detach().cpu().unsqueeze(1)),

            get_scar_ratio_cont(preds[i]),
            get_scar_ratio_cont(gt[i].detach().cpu().unsqueeze(1)),

            get_label_cont_std(preds[i]),
            get_label_cont_std(gt[i].detach().cpu().unsqueeze(1)),

            get_surface2vol_ratio(preds[i], 1.)[2],
            get_surface2vol_ratio(gt[i].detach().cpu().unsqueeze(1), 1.)[2],
            get_surface2vol_ratio(preds[i], 2.)[2],
            get_surface2vol_ratio(gt[i].detach().cpu().unsqueeze(1), 2.)[2],
            get_surface2vol_ratio(preds[i], 3.)[2],
            get_surface2vol_ratio(gt[i].detach().cpu().unsqueeze(1), 3.)[2],
            f"{network_name}"]
    
    df.to_csv(os.path.join(folder,"metrics.csv"))
    logger.info("Results Saved Successfully!!")    

def get_factor(network:str)->float:
    if network == "unet":
        return 3.
    elif network == "unet_bilstm":
        return 6.
    elif network == "unet_multi_bilstm":
        return 9.
    elif "tcm" in network:
        return 12.


if __name__ == '__main__':

    args = get_args()
    root = r"C:\Users\JAVIER\OneDrive\Escritorio\ETH\Year 2\Spring 2023\Semester Project\scarnetwork-pytorch"
    path = os.path.join(root, args.results_folder)
    logger = logger_setup(folder=os.path.join(root, args.results_folder, "volumes"))

    with open(os.path.join(path, "config.txt")) as f:
        config = f.read()

    config = config.split(r", ")
    config[1] = r"device=0"
    del config[2]
    if not any(["normalize" in arg for arg in config]): 
        config.append(config[-1])
        config[-2] = "normalize=False"
    if not any(["n_slices_central" in arg for arg in config]): 
        config.append(config[-1])
        config[-2] = "n_slices_central=-1"
    if not any(["n_slices_window" in arg for arg in config]): 
        config.append(config[-1])
        config[-2] = "n_slices_window=-1"
    config = r", ".join(config)
    config = eval(config)
    if "tcm" not in config.network:
        seed_everything(config.seed)

    partition = config.fixed_sets

    os.chdir(root)
    predictor = Predictor(fixed_sets=partition, 
                          cohorts=all_cohorts, 
                          logger=logger,
                          config=config)
    
    predictor.load_model_weights(os.path.join(path, "network"))
    factor = get_factor(config.network)
    logger.info(f"Applied factor: {factor} -> myo={factor+1}, sd2/fwhm={factor+2}, sd5={factor+3}")

    for mode in ["test", "val", "train"]:
        logger.info(f"***** DATASET: {mode} *****")
        img_list, preds_list, gt_list, ids_list, __ = predictor.predict_all(dataset=mode)
        
        save_volumes_and_table(gt=gt_list, 
                               preds=preds_list, 
                               pids=ids_list, 
                               network_name=config.network, 
                               folder=os.path.join(path, "volumes", mode),
                               factor=factor)



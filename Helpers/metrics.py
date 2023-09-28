
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def recall(TP, TN, FP, FN):
    return TP/(TP + FN)

def precision(TP, TN, FP, FN):
    return TP/(TP + FP)

def f1_score(TP, TN, FP, FN):
    return TP/(TP + 0.5*FP + FN)

def dice_score(TP, TN, FP, FN):
    return (2*TP) / (2*TP + FN + FP)

def jaccard_index(TP, TN, FP, FN):
    return (TP) / (TP + FP + FN) 

def extract_values(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    return TP, TN, FP, FN



def get_cm(cm, use):
    if use == "myo_sd":
        return get_cm_myo_sd(cm)
    elif use == "myo_fwhm":
        return get_cm_myo_fwhm(cm)
    elif use == "scar_2sd":
        return get_cm_scar_2sd(cm)
    elif use == "scar_5sd":
        return get_cm_scar_5sd(cm)
    elif use == "scar_fwhm":
        return get_cm_scar_fwhm(cm)
    
def get_cm_scar_fwhm(cm):
    cm = cm.to('cpu')
    cm[0] = cm[0] + cm[1]
    cm[:,0] = cm[:,0] + cm[:,1]
    cm = torch.Tensor([[cm[0,0].item(), cm[0,2].item()], [cm[2,0].item(), cm[2,2].item()]])
    return extract_values(cm)
    
def get_cm_scar_5sd(cm):
    cm= cm.to('cpu')
    cm[0] = cm[0] + cm[1] + cm[2]
    cm[:,0] = cm[:,0] + cm[:,1] + cm[:,2]
    cm = torch.Tensor([[cm[0,0].item(), cm[0,3].item()], [cm[3,0].item(), cm[3,3].item()]])
    return extract_values(cm)
    
def get_cm_scar_2sd(cm):
    cm = cm.to('cpu')
    cm[0] = cm[0] + cm[1]
    cm[2] = cm[2] + cm[3]
    cm[:,0] = cm[:,0] + cm[:,1]
    cm[:,2] = cm[:,2] + cm[:,3]
    cm = torch.Tensor([[cm[0,0].item(), cm[0,2].item()], [cm[2,0].item(), cm[2,2].item()]])
    return extract_values(cm)
    
def get_cm_myo_fwhm(cm):
    cm = cm.to('cpu')
    cm[1] = cm[1] + cm[2]
    cm[:,1] = cm[:,1] + cm[:,2]
    cm = cm[0:2, 0:2]     
    return extract_values(cm)                    
                            

def get_cm_myo_sd(cm:list):
    cm = cm.to('cpu')
    cm[1] = cm[1] + cm[2] + cm[3]
    cm[:,1] = cm[:,1] + cm[:,2] + cm[:,3]
    cm = cm[0:2, 0:2]

    return extract_values(cm)
    


def plot_training_scores(cms_train:list, cms_val:list, title:str, use:str, best_epoch=None):

    train = {'accuracy': [],
            'recall': [],
            'precision': [],
            'f1_score': [],
            'dice_score': [],
            'jaccard_index': []
            }

    val = {'accuracy': [],
            'recall': [],
            'precision': [],
            'f1_score': [],
            'dice_score': [],
            'jaccard_index': []
            }

    for cm_epoch in cms_train:
        TP, TN, FP, FN = get_cm(cm_epoch, use)  
        
        train['accuracy'].append(accuracy(TP, TN, FP, FN))
        train['recall'].append(recall(TP, TN, FP, FN))
        train['precision'].append(precision(TP, TN, FP, FN))
        train['f1_score'].append(f1_score(TP, TN, FP, FN))
        train['dice_score'].append(dice_score(TP, TN, FP, FN))
        train['jaccard_index'].append(jaccard_index(TP, TN, FP, FN))
        
    for cm_epoch in cms_val:
        TP, TN, FP, FN = get_cm(cm_epoch, use)                        
        
        val['accuracy'].append(accuracy(TP, TN, FP, FN))
        val['recall'].append(recall(TP, TN, FP, FN))
        val['precision'].append(precision(TP, TN, FP, FN))
        val['f1_score'].append(f1_score(TP, TN, FP, FN))
        val['dice_score'].append(dice_score(TP, TN, FP, FN))
        val['jaccard_index'].append(jaccard_index(TP, TN, FP, FN))
        
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs[0, 0].plot(train['accuracy'], color='green', label='train')
    axs[0, 0].plot(val['accuracy'], color='orange', label='val')
    if best_epoch is not None:
         axs[0, 0].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"Accuracy at best epoch:      train - {train['accuracy'][best_epoch]:.4f} | val - {val['accuracy'][best_epoch]:.4f}")
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].legend()

    axs[0, 1].plot(train['recall'], color='green', label='train')
    axs[0, 1].plot(val['recall'], color='orange', label='val')
    if best_epoch is not None:
         axs[0, 1].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"Recall at best epoch:        train - {train['recall'][best_epoch]:.4f} | val - {val['recall'][best_epoch]:.4f}")
    axs[0, 1].set_title('Recall')
    axs[0, 1].legend()

    axs[1, 0].plot(train['precision'], color='green', label='train')
    axs[1, 0].plot(val['precision'], color='orange', label='val')
    if best_epoch is not None:
         axs[1, 0].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"Precision at best epoch:     train - {train['precision'][best_epoch]:.4f} | val - {val['precision'][best_epoch]:.4f}")
    axs[1, 0].set_title('Precision')
    axs[1, 0].legend()

    axs[1, 1].plot(train['f1_score'], color='green', label='train')
    axs[1, 1].plot(val['f1_score'], color='orange', label='val')
    if best_epoch is not None:
         axs[1, 1].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"F1 Score at best epoch:      train - {train['f1_score'][best_epoch]:.4f} | val - {val['f1_score'][best_epoch]:.4f}")
    axs[1, 1].set_title('F1 Score')
    axs[1, 1].legend()

    axs[2, 0].plot(train['dice_score'], color='green', label='train')
    axs[2, 0].plot(val['dice_score'], color='orange', label='val')
    if best_epoch is not None:
         axs[2, 0].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"Dice Score at best epoch:    train - {train['dice_score'][best_epoch]:.4f} | val - {val['dice_score'][best_epoch]:.4f}")
    axs[2, 0].set_title('Dice Score')
    axs[2, 0].legend()

    axs[2, 1].plot(train['jaccard_index'], color='green', label='train')
    axs[2, 1].plot(val['jaccard_index'], color='orange', label='val')
    if best_epoch is not None:
         axs[2, 1].vlines(best_epoch, ymin=0, ymax=+1, linestyles="-.", label="best epoch")
         print(f"Jaccard Index at best epoch: train - {train['jaccard_index'][best_epoch]:.4f} | val - {val['jaccard_index'][best_epoch]:.4f}")
    axs[2, 1].set_title('Jaccard Index')
    axs[2, 1].legend()

    fig.suptitle(title)


def plot_loss(loss_train, loss_val, title, best_epoch=None):
    plt.plot(loss_train, color='green', label='train')
    plt.plot(loss_val, color='orange', label='val')
    if best_epoch is not None:
        plt.vlines(best_epoch, ymin=-1, ymax=+2, linestyles="-.", label="best epoch")
    plt.xlabel('epoch')
    plt.ylabel('dice loss')
    plt.yscale("log")
    plt.ylim([0.9*min(loss_train), 1.1*max(loss_val)])
    plt.legend()
    plt.title(title)


def get_rgb(im:torch.Tensor):
    new_im = np.zeros(tuple(im.size()) + (3,))
    im = im.numpy()

    # Green: Myo
    green = (im == 1.) * 1.
    # Blue: SD2 or FWHM
    blue = (im == 2.) * 1.
    # Red: SD5
    red = (im == 3.) * 1.

    new_im[..., 0] = red
    new_im[..., 1] = green
    new_im[..., 2] = blue
    
    return new_im
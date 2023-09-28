import os
import logging
from argparse import Namespace

import numpy as np
import time
from datetime import datetime

from matplotlib import pyplot as plt

import torch
from torch import optim
from torchmetrics import ConfusionMatrix
import segmentation_models_pytorch as smp

from Helpers.utils import get_exp_num, parse_json_file
from Helpers.data_utils import load_partitioned_data, load_all_data, train_val_test_split, preprocessing
from Normalization.nyul_normalizer import Normalizer_Nyul

def get_dice(confusion_matrix, channel):
    TP = torch.sum(confusion_matrix[channel:, channel:])
    FN_FP = torch.sum(confusion_matrix) - torch.sum(confusion_matrix[0:channel,0:channel]) - TP
    dice = (2*TP)/(2*TP + FN_FP)
    return dice

def save_plot(train, val, title, output_path):
    plt.plot(train, color='green', label='train')
    plt.plot(val, color='orange', label='val')
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


class BaseTrainer:
    def __init__(self, 
                 fixed_sets:str,
                 cohorts:list, 
                 config:Namespace, 
                 logger:logging.Logger):

        self._set_config(config)
        self.logger = logger

        self.output_folder = self._get_output_folder()
        self.normalizer = self._get_normalizer()

        self.net = self._get_model().to(self.config.device) 
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.config.lr)
        self.logger.info(f"Using optimzer:\n{self.optimizer}")

        self.criterion = smp.losses.DiceLoss('multiclass', classes=[i for i in range(1, self.config.n_channels)], from_logits=True)

        train_X, train_Y, val_X, val_Y, test_X, test_Y = self._get_data(fixed_sets, cohorts)

        self.trainloader = self._get_loader(train_X, train_Y, mode="train")
        self.validloader = self._get_loader(val_X, val_Y, mode="val")
        self.testloader =  self._get_loader(test_X, test_Y, mode="test")

        self.net.train()

    def train(self):

        self._initialize_metrics()
        start = time.time()
        while (self.early_stopping_counter < self.config.early_stopping) & (self.epoch <= self.config.max_epochs):
            self.logger.info('---------------- Epoch: {} ---------------------'.format(self.epoch))

            self.train_epoch()
            self.test(mode="val")

            self._print_epoch_results()
            self._update_epoch()
        
        self._terminate_train()
        finished = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        self.logger.info(f"TRAINING FINISHED. It took {finished}.")


    def _set_config(self, config):

        config.device = torch.device('cuda:' + str(config.device) if torch.cuda.is_available() else "cpu")
    
        if config.label_method == 'FWHM':
            config.n_channels = 3
        else:
            config.n_channels = 4

        config.n_classes = config.n_channels

        if config.n_slices_central != -1 and config.n_slices_window != -1:
            raise ValueError("One of --n_slices_central, n_slices_window must be equal to -1.")

        if config.views == '3D':
            config.output_folder = os.path.join(config.output_folder, "3D")
            if config.n_slices_central != -1: 
                config.bs = config.bs // config.n_slices_central
            elif config.n_slices_window != -1:
                config.bs = config.bs // config.n_slices_window
        elif config.views == '2D':
            config.output_folder = os.path.join(config.output_folder, "2D")

        self.config = config
    
    
    def _get_output_folder(self,):

        if self.config.n_cross_val is not None:
            cv_idx = self.config.fixed_sets.split(".json")[0].split("_")[-1]
            exp_name =  datetime.now().strftime("%Y_%m_%d_") + get_exp_num(self.config.output_folder, int(cv_idx)>0) + '_epochs' + str(self.config.max_epochs) + '_lr' + str(self.config.lr) + '_' + self.config.network + '_' + self.config.label_method 

            experiment_dir = os.path.join(self.config.output_folder, exp_name+"_cv", f"cv_{cv_idx}")
            paritition = parse_json_file(self.config.fixed_sets)
            train_samples = ["_".join(p.split(os.sep)[-2:]) for p in paritition["train"]]
            valid_samples = ["_".join(p.split(os.sep)[-2:]) for p in paritition["val"]]
            test_samples =  ["_".join(p.split(os.sep)[-2:]) for p in paritition["test"]]
            
            self.logger.info(f"Train Samples: {train_samples}")
            self.logger.info(f"Valid Samples: {valid_samples}")
            self.logger.info(f"Test  Samples: {test_samples}")
        else:
            exp_name =  datetime.now().strftime("%Y_%m_%d_") + get_exp_num(self.config.output_folder, False) + '_epochs' + str(self.config.max_epochs) + '_lr' + str(self.config.lr) + '_' + self.config.network + '_' + self.config.label_method 
            experiment_dir = os.path.join(self.config.output_folder, exp_name)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        
        
        return experiment_dir
    
    def _get_normalizer(self):
        if self.config.normalize:
            return Normalizer_Nyul()
        else:
            return None    

    def _get_data(self, fixed_sets:str, cohorts:list) -> tuple:

        fix_order = True if self.config.views=="3D" else False
        
        if self.config.use_predifined_sets:
            self.logger.info(f"Using dataset partition from {fixed_sets}")
            train_data, valid_data, test_data = load_partitioned_data(fixed_sets, method=self.config.label_method, fix_order=fix_order)
        else:
            self.logger.info(f"Using random data partition.")
            images, labels, pids = load_all_data(cohorts, self.config.data_folder, method=self.config.label_method, fix_order=fix_order)
            train_data, valid_data, test_data = train_val_test_split(images, labels, pids)

        return train_data, valid_data, test_data


    def _initialize_metrics(self):

        # list with epoch values
        self.train_scores = []
        self.val_scores = []

        self.train_loss = []
        self.val_loss = []

        self.train_myo_dice = []
        self.val_myo_dice = []

        if self.config.label_method == "SD":
            self.train_sd2_dice = []
            self.train_sd5_dice = []
            self.val_sd2_dice = []
            self.val_sd5_dice = []
        elif self.config.label_method == "FWHM":
            self.train_fwhm_dice = []
            self.val_fwhm_dice = []

        # best scores
        self.best_net = None
        self.train_cache_loss = 100
        self.val_cache_loss = 100
        self.best_loss = 100
        self.best_train_loss = 100
        self.epoch = 1
        self.early_stopping_counter = 0

        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.config.n_channels).to(self.config.device)


    def _update_metrics(self, confusion_matrix, batch_losses, mode):

        loss = np.round(np.mean(batch_losses), 4)

        if mode == "train":
            self.train_cache_loss = loss
            self.train_loss.append(loss)
            self.train_scores.append(confusion_matrix)

            self.train_myo_dice.append(get_dice(confusion_matrix, 1).item())
            if self.config.label_method == "FWHM":
                self.train_fwhm_dice.append(get_dice(confusion_matrix, 2).item())
            if self.config.label_method == "SD":
                self.train_sd2_dice.append(get_dice(confusion_matrix, 2).item())
                self.train_sd5_dice.append(get_dice(confusion_matrix, 3).item())
        
        elif mode == "val":
            self.val_cache_loss = loss
            self.val_loss.append(loss)
            self.val_scores.append(confusion_matrix)
            
            self.val_myo_dice.append(get_dice(confusion_matrix, 1).item())
            if self.config.label_method == "FWHM":
                self.val_fwhm_dice.append(get_dice(confusion_matrix, 2).item())
            if self.config.label_method == "SD":
                self.val_sd2_dice.append(get_dice(confusion_matrix, 2).item())
                self.val_sd5_dice.append(get_dice(confusion_matrix, 3).item())

        elif mode == "test":
            self.test_loss = loss
            self.test_scores = confusion_matrix
            
            self.test_myo_dice = get_dice(confusion_matrix, 1).item()
            if self.config.label_method == "FWHM":
                self.test_fwhm_dice = get_dice(confusion_matrix, 2).item()
            if self.config.label_method == "SD":
                self.test_sd2_dice = get_dice(confusion_matrix, 2).item()
                self.test_sd5_dice = get_dice(confusion_matrix, 3).item()


    def _update_epoch(self):

        if self.val_cache_loss < self.best_loss:
            self.best_loss = self.val_cache_loss
            self.best_train_loss = self.train_cache_loss
            self.best_net = self.net
            torch.save(self.net.state_dict(), os.path.join(self.output_folder, 'network'))
            self.early_stopping_counter = 0

            self.best_myo_dice = self.val_myo_dice[-1]
            self.best_train_myo_dice = self.train_myo_dice[-1]
            if self.config.label_method == 'FWHM':
                self.best_fwhm_dice = self.val_fwhm_dice[-1]
                self.best_train_fwhm_dice = self.train_fwhm_dice[-1]
            if self.config.label_method == 'SD':
                self.best_sd2_dice = self.val_sd2_dice[-1]
                self.best_sd5_dice = self.val_sd5_dice[-1]
                self.best_train_sd2_dice = self.train_sd2_dice[-1]
                self.best_train_sd5_dice = self.train_sd5_dice[-1]
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.config.early_stopping: self.logger.info(f"Early Stopping of Training. Loss not improved in {self.config.early_stopping} epochs.")
        
        self.epoch += 1

    
    def _print_epoch_results(self):
        if self.config.label_method == "FWHM":
            self.logger.info('epoch {} ---- train loss {} ---- train myo dice {} ---- train FWHM dice {}'.format(self.epoch, self.train_loss[-1], np.round(self.train_myo_dice[-1], 4), 
                                                                                                              np.round(self.train_fwhm_dice[-1], 4)))
            self.logger.info('epoch {} ---- val loss {} ---- val myo dice {} ---- val FWHM dice {}'.format(self.epoch, self.val_loss[-1], np.round(self.val_myo_dice[-1], 4), 
                                                                                                np.round(self.val_fwhm_dice[-1], 4)))

        elif self.config.label_method == "SD":
            self.logger.info('epoch {} ---- train loss {} ---- train myo dice {} ---- train 2-SD dice {} ---- train 5-SD dice {}'.format(self.epoch, self.train_loss[-1], 
                                                                                                                                      np.round(self.train_myo_dice[-1], 4),
                                                                                                                                      np.round(self.train_sd2_dice[-1], 4), 
                                                                                                                                      np.round(self.train_sd5_dice[-1], 4)))
            self.logger.info('epoch {} ---- val loss {} ---- val myo dice {} ---- val 2-SD dice {} ---- val 5-SD dice {}'.format(self.epoch, self.val_loss[-1], 
                                                                                                                        np.round(self.val_myo_dice[-1], 4), 
                                                                                                                        np.round(self.val_sd2_dice[-1], 4), 
                                                                                                                        np.round(self.val_sd5_dice[-1], 4)))


    def _terminate_train(self):

        self.net = self.best_net

        torch.save(self.train_scores, os.path.join(self.output_folder, 'train_scores.pt'))
        torch.save(self.val_scores, os.path.join(self.output_folder, 'val_scores.pt'))
        torch.save(self.train_loss, os.path.join(self.output_folder, 'train_loss.pt'))
        torch.save(self.val_loss, os.path.join(self.output_folder, 'val_loss.pt'))

        with open(os.path.join(self.output_folder, 'config.txt'), 'w') as c:
            c.write(str(self.config))

        save_plot(self.train_loss, self.val_loss, "Loss - Dice Loss", os.path.join(self.output_folder, "dice_loss.png"))
        save_plot(self.train_myo_dice, self.val_myo_dice, "Dice Score Myocardium", os.path.join(self.output_folder, "myo_dice.png"))
        if self.config.label_method == "FWHM":
            save_plot(self.train_fwhm_dice, self.val_fwhm_dice, "Dice Score", os.path.join(self.output_folder, "scar_fwhm_dice.png"))
            self.logger.info('epochs {} -- train loss {} -- val loss {} -- train myo dice {} -- val myo dice {} -- train fwhm dice {} -- val fwhm dice {}'.format(self.epoch - self.config.early_stopping, 
                                                                                                                                                                  self.best_train_loss, 
                                                                                                                                                                  self.best_loss,
                                                                                                                                                                  np.round(self.best_train_myo_dice, 4), 
                                                                                                                                                                  np.round(self.best_myo_dice, 4),
                                                                                                                                                                  np.round(self.best_train_fwhm_dice, 4),
                                                                                                                                                                  np.round(self.best_fwhm_dice, 4)))
        elif self.config.label_method == "SD":
            save_plot(self.train_sd2_dice, self.val_sd2_dice, "Dice Score", os.path.join(self.output_folder, "scar_sd2_dice.png"))
            save_plot(self.train_sd5_dice, self.val_sd5_dice, "Dice Score", os.path.join(self.output_folder, "scar_sd5_dice.png"))
            self.logger.info('epochs {} -- train loss {} -- val loss {} -- train myo dice {} -- val myo dice {} -- train 2-SD dice {} -- val 2-SD dice {} -- train 5-SD dice {} -- val 5-SD dice {}'.format(self.epoch - self.config.early_stopping, 
                                                                                                                                                                                                            self.best_train_loss, 
                                                                                                                                                                                                            self.best_loss,
                                                                                                                                                                                                            np.round(self.best_train_myo_dice, 4), 
                                                                                                                                                                                                            np.round(self.best_myo_dice, 4),
                                                                                                                                                                                                            np.round(self.best_train_sd2_dice, 4),
                                                                                                                                                                                                            np.round(self.best_sd2_dice, 4),
                                                                                                                                                                                                            np.round(self.best_train_sd5_dice, 4),
                                                                                                                                                                                                            np.round(self.best_sd5_dice, 4)))


    def _terminate_test(self):
        if self.config.label_method == "FWHM":
            self.logger.info('test loss {} -- test myo dice {} -- test fwhm dice {}'.format(self.test_loss, 
                                                                            np.round(self.test_myo_dice, 4), 
                                                                            np.round(self.test_fwhm_dice, 4)))
        elif self.config.label_method == "SD":
            self.logger.info('test loss {} -- test myo dice {} -- test 2-SD dice {} -- test 5-SD dice {}'.format(self.test_loss, 
                                                                                                np.round(self.test_myo_dice, 4), 
                                                                                                np.round(self.test_sd2_dice, 4),
                                                                                                np.round(self.test_sd5_dice, 4)))
        torch.save(self.test_scores, os.path.join(self.output_folder, 'test_scores.pt'))


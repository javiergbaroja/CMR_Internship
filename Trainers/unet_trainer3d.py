import os
import logging

from tqdm import tqdm
import importlib
from argparse import Namespace

import torch
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from Helpers.TestTimeAugmentation import predictTTA
from Helpers.PerPatientDataset import get_ImageDataset3D
from Helpers.utils import seed_worker
from Helpers.data_utils import preprocessing

from .base_trainer import BaseTrainer


class Trainer3D(BaseTrainer):
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.46, patience=self.config.early_stopping//3) #
        self.logger.info(f"Using optimzer:\n{self.optimizer}")

        self.criterion = smp.losses.DiceLoss('multiclass', classes=[i for i in range(1, self.config.n_channels)], from_logits=True)

        train_data, valid_data, test_data = self._get_data(fixed_sets, cohorts)

        self.trainloader = self._get_loader(train_data, mode="train")
        self.validloader = self._get_loader(valid_data, mode="val")
        self.testloader =  self._get_loader(test_data, mode="test")

        self.net.train()


    def train_epoch(self):

        confusion_matrix = 0
        batch_losses = []
        self.net.train()
        with tqdm(self.trainloader, unit='batch') as batch_dataloader:
            for batch in batch_dataloader:
                batch_dataloader.set_description(f'Epoch {self.epoch}')

                imgs, masks_gt = self._extract_batch(batch)
                masks_pred = self.net(imgs, vars(self.config))

                final_masks_pred = torch.argmax(masks_pred, dim=1)
                confusion_matrix += self.confmat(final_masks_pred.flatten(), masks_gt.flatten())
        
                # calculate loss
                loss = self.criterion(masks_pred, masks_gt.long())
                loss = loss.mean()
                batch_losses.append(loss.item())
                batch_dataloader.set_postfix(loss=loss.item())
                
                # optimizer update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

        self._update_metrics(confusion_matrix, batch_losses, "train")


    def test(self, mode:str):

        self.net.eval()

        confusion_matrix = 0
        batch_losses = []

        dataloader = self.validloader if mode == "val" else self.testloader
        with torch.no_grad():
            with tqdm(dataloader, unit='batch') as batch_dataloader:
                for batch in batch_dataloader:
                    batch_dataloader.set_description(f'Epoch {self.epoch}')

                    imgs, masks_gt = self._extract_batch(batch)
                    
                    if self.config.use_tta:
                        masks_pred = predictTTA(self.net, imgs, vars(self.config))
                    else:
                        masks_pred = self.net(imgs, vars(self.config))

                    final_masks_pred = torch.argmax(masks_pred, dim=1)
                    confusion_matrix += self.confmat(final_masks_pred.flatten(), masks_gt.flatten())
            
                    # calculate loss
                    loss = self.criterion(masks_pred, masks_gt.long())
                    loss = loss.mean()
                    batch_losses.append(loss.item())
                    batch_dataloader.set_postfix(loss=loss.item())

        self._update_metrics(confusion_matrix, batch_losses, mode)

        if mode=="val":
            self.scheduler.step(self.val_loss[-1])
        elif mode=="test":
            self._terminate_test()

    
    def _extract_batch(self, batch):

        imgs = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['image']]
        masks_gt = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['mask']]
        masks_gt = torch.round(torch.cat(masks_gt, dim=0))[:,0]

        return imgs, masks_gt
    
    
    def _get_model(self):

        m = importlib.import_module(f'Models.{self.config.network}')
        model = getattr(m, "ModelLoader")(**vars(self.config))
        
        self.logger.info(f"{self.config.views} Model type chosen with architecture {self.config.network} and {self.config.encoder} encoder.")
        self.logger.info(f"Architecture Summary:\n{model}")
        
        return model


    def _get_loader(self, data:dict, mode:str) -> DataLoader:

        if self.config.n_slices_central != -1 or self.config.n_slices_window != -1: # All the patients have the same number of slices
            even_dataset = True 
            bs = self.config.bs
        else: # All slices are loaded for each patient
            even_dataset = False
            bs = 1

        if mode == "train":
            data = preprocessing(data, self.config.views, self.config.n_slices_central, self.config.n_slices_window)

            if self.normalizer is not None:
                data["images"] = self.normalizer.nyul_train_and_normalize(data["images"])

            dataloader = DataLoader(get_ImageDataset3D(data=data, 
                                                       even_dataset=even_dataset, 
                                                       augment=self.config.augment, 
                                                       batch_size=self.config.bs), 
                                    batch_size=bs, 
                                    shuffle=True, # Double check this
                                    worker_init_fn=seed_worker, 
                                    generator=torch.Generator().manual_seed(self.config.seed), 
                                    num_workers=4)
        else: # For val and test use all the image data -> does not drop any slices.
            if self.config.all_test_slices:
                n_central = -1
                n_window = -1
            else:
                n_central = self.config.n_slices_central
                n_window = self.config.n_slices_window

            data = preprocessing(data, self.config.views, n_central, n_window)

            if self.normalizer is not None:
                data["images"] = self.normalizer.hist_normalize(data["images"])
                
            dataloader = DataLoader(get_ImageDataset3D(data=data,
                                                       even_dataset=False, 
                                                       batch_size=-1), 
                                    batch_size=1, 
                                    shuffle=False)
        return dataloader


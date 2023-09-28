
import logging

from tqdm import tqdm

from argparse import Namespace

import torch
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from Helpers.TestTimeAugmentation import predictTTA
from Helpers.PerPatientDataset import byPatientDataset3D
from Helpers.utils import seed_worker
from Helpers.data_utils import preprocessing

from .base_trainer import BaseTrainer


class Trainer2D(BaseTrainer):
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
                masks_pred = self.net(imgs)

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
                        masks_pred = predictTTA(self.net, imgs)
                    else:
                        masks_pred = self.net(imgs)

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

        imgs = batch['image'].to(device=self.config.device, dtype=torch.float32)
        masks_gt = batch['mask'].to(device=self.config.device, dtype=torch.float32)
        masks_gt = torch.round(masks_gt)[:,0]

        return imgs, masks_gt
    
    
    def _get_model(self):

        model = smp.Unet(
            encoder_name="resnet34",  
            encoder_weights="imagenet",  
            in_channels=1, 
            classes=self.config.n_channels,  
            activation=None).to(self.config.device)
        
        self.logger.info(f"{self.config.views} Model type chosen with architecture {self.config.network} and {self.config.encoder} encoder.")
        self.logger.info(f"Architecture Summary:\n{model}")
        
        return model


    def _get_loader(self, data:dict, mode:str) -> DataLoader:

        data = preprocessing(data, self.config.views)

        if self.normalizer is not None:
            data["images"] = self.normalizer.hist_normalize(data["images"]) if mode != "train" else self.normalizer.nyul_train_and_normalize(data["images"])

        if mode == "train":
            dataloader = DataLoader(byPatientDataset3D(data, augment=self.config.augment), 
                                    batch_size=self.config.bs, 
                                    shuffle=True, 
                                    worker_init_fn=seed_worker, 
                                    generator=torch.Generator().manual_seed(self.config.seed), 
                                    num_workers=4)
        else: 
            dataloader = DataLoader(byPatientDataset3D(data), 
                                    batch_size=self.config.bs, 
                                    shuffle=False)
        
        return dataloader


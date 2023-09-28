from .unet_trainer3d import Trainer3D
import logging
import segmentation_models_pytorch as smp
import argparse
from Helpers.data_utils import preprocessing, load_all_data, load_partitioned_data, train_val_test_split
from Helpers.utils import seed_worker
from Helpers.PerPatientDataset import get_ImageDataset3D
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torchmetrics import ConfusionMatrix

def get_dice(confusion_matrix, channel):
    TP = torch.sum(confusion_matrix[channel:, channel:])
    FN_FP = torch.sum(confusion_matrix) - torch.sum(confusion_matrix[0:channel,0:channel]) - TP
    dice = (2*TP)/(2*TP + FN_FP)
    return dice

class Predictor(Trainer3D):
    def __init__(self, 
                 fixed_sets: str, 
                 cohorts: list, 
                 config: argparse.Namespace, 
                 logger: logging.Logger):
        
        self._set_config(config)
        self.logger = logger
        self.normalizer = self._get_normalizer()
        self.net = self._get_model().to(self.config.device) if self.config.views == "3D" else self._get_model_2d()

        train_data, valid_data, test_data = self._get_data(fixed_sets, cohorts)

        self.trainloader = self._get_loader(train_data, mode="train")
        self.validloader = self._get_loader(valid_data, mode="val")
        self.testloader =  self._get_loader(test_data, mode="test")
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.config.n_channels).to(self.config.device)

    def _get_model_2d(self):

        model = smp.Unet(
            encoder_name="resnet34",  
            encoder_weights="imagenet",  
            in_channels=1, 
            classes=self.config.n_channels,  
            activation=None).to(self.config.device)
        
        self.logger.info(f"{self.config.views} Model type chosen with architecture {self.config.network} and {self.config.encoder} encoder.")
        self.logger.info(f"Architecture Summary:\n{model}")
        
        return model


    def _get_data(self, fixed_sets:str, cohorts:list) -> tuple:
      
        if self.config.use_predifined_sets:
            self.logger.info(f"Using dataset partition from {fixed_sets}")
            train_data, valid_data, test_data = load_partitioned_data(fixed_sets, method=self.config.label_method, fix_order=True)
        else:
            self.logger.info(f"Using random data partition.")
            images, labels, pids = load_all_data(cohorts, self.config.data_folder, method=self.config.label_method, fix_order=True)
            train_data, valid_data, test_data = train_val_test_split(images, labels, pids)

        return train_data, valid_data, test_data

    def _get_loader(self, data, mode:str) -> DataLoader:

        data = preprocessing(data,self.config.views, -1, -1)
        
        if self.normalizer is not None:
            data["images"] = self.normalizer.hist_normalize(data["images"]) if mode != "train" else self.normalizer.nyul_train_and_normalize(data["images"])

        dataloader = DataLoader(get_ImageDataset3D(data=data, 
                                                   augment=False,
                                                   even_dataset=False, 
                                                   batch_size=-1), 
                                batch_size=1, 
                                shuffle=False)
        return dataloader
    
    def load_model_weights(self, state_dict:str):
        self.logger.info(f"Loading model weights from: {state_dict}")
        self.net.load_state_dict(torch.load(state_dict, map_location=self.config.device))
        self.net.eval()

    def predict_all(self, dataset:str):

        self.net.eval()

        if dataset == "train":
            dataloader = self.trainloader
        elif dataset == "test":
            dataloader = self.testloader
        elif dataset == "val":
            dataloader = self.validloader

        img_list, preds_list, gt_list, pIDs = [], [], [], []

        with torch.no_grad():
            with tqdm(dataloader, unit='batch') as batch_dataloader:
                for batch in batch_dataloader:
                    batch_dataloader.set_description('Predict set')

                    imgs, masks_gt = self._extract_batch(batch)

                    if self.config.views=="3D":
                        masks_pred = self.net(imgs, vars(self.config))
                    elif self.config.views=="2D":
                        masks_pred = self.net(imgs[0].unsqueeze(0))

                    final_masks_pred = torch.unsqueeze(torch.argmax(masks_pred, dim=1), 1)
                    # confusion_matrix = self.confmat(final_masks_pred.detach().flatten(), masks_gt.to(self.config.device).flatten())
                    img_list.append(imgs[0].detach().cpu())
                    preds_list.append(final_masks_pred.detach().cpu())
                    gt_list.append(masks_gt.detach().cpu())
                    pIDs.append(batch["pID"])

        if self.config.views == "3D":
            pIDs = [id[0][0] for id in pIDs]
        else:
            img_list, preds_list, gt_list, pIDs = self._prepare_output(img_list, preds_list, gt_list, pIDs)
        dice_scores = self._calc_dice_score(preds_list, gt_list)
        return img_list, preds_list, gt_list, pIDs, dice_scores 
    
    def _calc_dice_score(self, preds_list, gt_list):
        if self.config.label_method == "FWHM": dice_scores = [[],[]]
        elif self.config.label_method == "SD": dice_scores = [[],[],[]]
        for i in range(len(preds_list)): 
            
            confusion_matrix = self.confmat(preds_list[i][:,0,:,:].flatten().to(self.config.device),   gt_list[i].detach().cpu().flatten().to(self.config.device))

            dice_scores[0].append(get_dice(confusion_matrix, 1).item())
            if self.config.label_method == "FWHM":
                dice_scores[1].append(get_dice(confusion_matrix, 2).item())
            if self.config.label_method == "SD":
                dice_scores[1].append(get_dice(confusion_matrix, 2).item())
                dice_scores[2].append(get_dice(confusion_matrix, 3).item())
        return dice_scores

    def _prepare_output(self, img_list, preds_list, gt_list, pIDs):

        img_list_new, preds_list_new, gt_list_new, pIDs_new = [],[],[],[]
        aux_imgs, aux_preds, aux_gts = None, None, None

        for i, pid in enumerate(pIDs):
            if i==0:
                aux_imgs = img_list[i]
                aux_preds = preds_list[i]
                aux_gts = gt_list[i]
            elif pid == pIDs[i-1]:
                aux_imgs = torch.cat((aux_imgs, img_list[i]), dim=0)
                aux_preds = torch.cat((aux_preds, preds_list[i]), dim=0)
                aux_gts = torch.cat((aux_gts, gt_list[i]), dim=0)
            else:
                img_list_new.append(aux_imgs.unsqueeze(1))
                preds_list_new.append(aux_preds)
                gt_list_new.append(aux_gts)
                pIDs_new.append(pIDs[i-1])
                aux_imgs = img_list[i]
                aux_preds = preds_list[i]
                aux_gts = gt_list[i]
                
        img_list_new.append(aux_imgs.unsqueeze(1))
        preds_list_new.append(aux_preds)
        gt_list_new.append(aux_gts)
        pIDs_new.append(pid)
        pIDs_new = [id[0][0] for id in pIDs_new]

        return img_list_new, preds_list_new, gt_list_new, pIDs_new
    

    def predict_one_sample(self, dataset:str, sample:int):

        self.net.eval()

        if dataset == "train":
            dataloader = iter(self.trainloader)
        elif dataset == "test":
            dataloader = iter(self.testloader)
        elif dataset == "val":
            dataloader = iter(self.validloader)

        img_list, mask_list = [], []

        with torch.no_grad():
            with tqdm(dataloader, unit='batch') as batch_dataloader:
                for batch in batch_dataloader:
                    batch_dataloader.set_description('Predict set')

                    imgs = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['image']]
                    masks_gt = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['mask']]
                    masks_gt = torch.round(torch.cat(masks_gt, dim=0))
                    

                    masks_pred = self.net(imgs, vars(self.config))

                    final_masks_pred = torch.unsqueeze(torch.argmax(masks_pred, dim=1), 1)
                    img_list.append(imgs[0])
                    mask_list.append(final_masks_pred)

        return img_list, mask_list
    
    def _extract_batch(self, batch):
        if self.config.views ==  "3D":
            imgs = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['image']]
            masks_gt = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['mask']]
            masks_gt = torch.round(torch.cat(masks_gt, dim=0))[:,0]
        elif self.config.views == "2D":
            imgs = [item.squeeze(0).to(device=self.config.device, dtype=torch.float32) for item in batch['image']]
            masks_gt = batch['mask'][0][0]
            masks_gt = torch.round(masks_gt)

        return imgs, masks_gt
            
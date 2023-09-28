import glob
import random
import os

from torch.utils.data import Dataset

import numpy as np
import elasticdeform

from scipy.interpolate import interp1d
from skimage import exposure
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import torch
import matplotlib.pyplot as plt

'''
Tom code data augmentation:
•	Brightness (0.8, 1.2)
•	PiecewiseAffine 0.02 mode = edge
•	Rotation
•	Shift 0.3
•	Shear range 0.3
•	Zoom range 0.3
•	Horizontal flip 
•	Vertical flip
'''

class ImageDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.augment = augment

        self.images = images
        self.masks = masks

        self.n = len(self.images)


    def transforms(self, image, mask):
        if random.random() > 0.5:
            vertical = T.RandomVerticalFlip(1)
            image = vertical(image)
            mask = vertical(mask)
        
        if random.random() > 0.5:
            horizontal = T.RandomHorizontalFlip(1)
            image = horizontal(image)
            mask = horizontal(mask)
            
        # brightness and contrast
        
        colorjitter = T.ColorJitter(brightness=0.2, contrast=0.2)
        image = colorjitter(image)
        
        # crop
        resize = T.Resize((image.shape[-2], image.shape[-1]))
        
        crop_by = random.randint(0,20)
        crop_size = (image.shape[-2]-crop_by, image.shape[-1]-crop_by)
        crop = T.CenterCrop(crop_size)
        image = crop(image)
        mask = crop(mask)
        image = resize(image)
        mask = resize(mask)
    
        return image, mask
    

    def __getitem__(self, index):

        img = self.images[index % self.n] + 0.
        msk = self.masks[index % self.n] + 0

        if self.augment:
            img, msk = self.transforms(img, msk)

        return {'image': img, 'mask': msk}


    def __len__(self):

        return self.n
    


"""
class ImageDataset(Dataset):

    def __init__(self, images, masks, augment=True, precrop=False, cropsize=192):

        self.images = []
        self.masks = []

        #load data:
        self.fromdisk = False

        self.augment = augment
        self.precrop = precrop
        self.cropsize = cropsize

        self.images = images
        self.masks = masks

        self.n = len(self.images)

        #in the elastic deformation, the intensity of the deformation depends on both sigma and points, we 
        #want to allow both to have a large range, but want to avoid simultaniously picking large values 
        #for both, so here we define a range of possible pairs, and then we sample from these pairs of values
        self.possible_point_sigma_combinations = []
        for i in range(2,11):
            for j in range(1,11):
                # if i < 5 or 3 < i+j < 12:
                if i < 5 or i+j < 10:
                    self.possible_point_sigma_combinations.append( (i,j) )

        self.elastic_zoom_min = 0.85
        self.elastic_zoom_max = 1.3
        self.elastic_rotation_limit = 180 #in degrees

        if not precrop:
            self.xpos = np.tile(np.linspace(0,1,self.images.shape[2])[:,None],(1,self.images.shape[3]))
            self.ypos = np.tile(np.linspace(0,1,self.images.shape[3])[None], (self.images.shape[2],1))
        else:
            self.xpos = np.tile(np.linspace(0,1,cropsize)[:,None],(1,cropsize))
            self.ypos = np.tile(np.linspace(0,1,cropsize)[None], (cropsize,1))

    def __getitem__(self, index):

        if self.fromdisk:
            img = np.load(self.images[index % self.n])
            msk = np.load(self.masks[index % self.n])
        else:
            img = self.images[index % self.n] + 0.
            msk = self.masks[index % self.n] + 0

        if self.precrop:
            s1 = np.random.randint(img.shape[1]-self.cropsize)
            s2 = np.random.randint(img.shape[2]-self.cropsize)
            img = img[:,s1:s1+self.cropsize,s2:s2+self.cropsize]
            msk = msk[:,s1:s1+self.cropsize,s2:s2+self.cropsize]

        if self.augment:
            if np.random.random() < 0.5:
                img = img[:,::-1]+0
                msk = msk[:,::-1]+0
            if np.random.random() < 0.5:
                img, msk = self.randomShapeAugmentation(img, msk)
            if np.random.random() < 0.7:
                img = self.randomIntensityAugmentation(img[0])[None]

        return {'image': img, 'mask': msk}

    def __len__(self):

        return self.n

    def randomShapeAugmentation(self, img, msk):

        choice = np.random.randint(len(self.possible_point_sigma_combinations))
        pts, sig = self.possible_point_sigma_combinations[choice]
        rot = (np.random.random()*2-1)*self.elastic_rotation_limit
        zoom = self.elastic_zoom_min + np.random.random()*(self.elastic_zoom_max-self.elastic_zoom_min)

        img, msk = elasticdeform.deform_random_grid( [img,msk], 
            sigma = sig, 
            points = pts,
            rotate = rot,
            zoom = zoom,
            mode=['mirror','constant'],
            axis=(1,2),
            order=[0,1])

        return img, msk

    def randomIntensityTransform(self, k=20, sort=True, w=0.2):
        '''
        returns a random intensity transformating function, i.e. a random function from [0,1]-->[0,1]
        this function is monotonicly increasing (if sort=True) and 'close to' x=y (if w is small)
        '''
        x = np.sort([0,1]+list(np.random.random(k)))
        y = x*(1-w)+np.random.random(k+2)*w
        if sort:
            y = np.sort(y)
        return interp1d(x, y, fill_value='extrapolate')

    def randomIntensityAugmentation(self, img):

        #with a 50% chance randomly mix the image with a random 'exposure adjusted' variant:
        choice = np.random.randint(0,10)
        if choice < 5:
            if choice == 0:
                p2, p98 = np.percentile(img, (5, 95))
                img2 = exposure.rescale_intensity(img, in_range=(p2, p98))
            elif choice == 1:
                img2 = exposure.equalize_hist(img)
            elif choice == 2:
                img2 = img
            # 	img2 = exposure.equalize_adapthist(img, clip_limit=0.03)
            elif choice == 3:
                img2 = img
                # img2 = exposure.adjust_log(img, gain=0.5+np.random.random(), inv=np.random.choice([True,False]))
            elif choice == 4:
                img2 = img
                # img2 = exposure.adjust_gamma(img, gamma=0.5+np.random.random(), gain=0.5+np.random.random())
            alpha = np.random.random()
            img = img*alpha+img2*(1-alpha)

        
        if np.random.random() < 0.5:
            #randomly trim off a little from the upper and lower intensity values:
            intensity_scale = (np.random.random()-0.5)*0.2
            img = np.clip(img * (1+intensity_scale) - np.random.random()*intensity_scale, 0, 1)

        #transform intensity based on random (peice-wise linear increasing) map from [0,1] to [0,1]:
        img = self.randomIntensityTransform(np.random.randint(3,25))(img)

        if np.random.random() < 0.5:
            #add a random gradient image (i.e for non global intensity variation, and *somewhat* resembling coil sensitivity variation:
            img += ((np.random.random()*2-1)*(self.xpos-np.random.random()) + (np.random.random()*2-1)*(self.ypos-np.random.random()))*0.3*np.random.random()

        #add Gaussian noise:
        img += np.random.randn(img.shape[0],img.shape[1])*0.03*np.random.random()
        img = np.clip(img,0,1)

        return img
"""
import nibabel as nib
from scipy.interpolate import interp1d
import numpy as np
import torch
from tqdm import tqdm

class Normalizer_Nyul():
    def __init__(self, 
                 i_min=0,
                 i_max=100,
                 i_s_min=0,
                 i_s_max=1,
                 l_percentile=10,
                 u_percentile=90,
                 step=50) -> None:
        self.i_min = i_min
        self.i_max = i_max
        self.i_s_min = i_s_min
        self.i_s_max = i_s_max
        self.l_percentile = l_percentile
        self.u_percentile = u_percentile
        self.step = step
        self.percs = np.concatenate(([self.i_min],
                                        np.arange(self.l_percentile, self.u_percentile+1, self.step),
                                        [self.i_max]))
        self.standard_scale = np.zeros(len(self.percs))

    def _get_landmarks(self, img):
        return np.percentile(img, self.percs)


    def nyul_train_standard_scale(self, all_imgs):

        # self.standard_scale = np.zeros(len(self.percs))

        # process each image in order to build the standard scale
        n_images = 0

        if isinstance(all_imgs, list):
            with tqdm(all_imgs, unit='volume') as images:
                for items in images:
                    for img in items:
                        landmarks = self._get_landmarks(img)
                        min_p = np.percentile(img, self.i_min)
                        max_p = np.percentile(img, self.i_max)
                        f = interp1d([min_p, max_p], [self.i_s_min, self.i_s_max])  # create interpolating function
                        landmarks = np.array(f(landmarks)) # interpolate landmarks
                        self.standard_scale += landmarks  # add landmark values of this volume to standard_scale
                        n_images += 1
        elif isinstance(all_imgs, torch.Tensor):
            with tqdm(all_imgs, unit='volume') as images:
                for img in images:
                    landmarks = self._get_landmarks(img)
                    min_p = np.percentile(img, self.i_min)
                    max_p = np.percentile(img, self.i_max)
                    f = interp1d([min_p, max_p], [self.i_s_min, self.i_s_max])  # create interpolating function
                    landmarks = np.array(f(landmarks)) # interpolate landmarks
                    self.standard_scale += landmarks  # add landmark values of this volume to standard_scale
                    n_images += 1

        self.standard_scale = self.standard_scale / n_images  # get mean values


    def _do_hist_normalization_img(self, 
                              input_image,
                              interp_type='linear'):

        landmarks = self._get_landmarks(input_image)
        f = interp1d(landmarks, self.standard_scale, kind=interp_type, fill_value='extrapolate')  # define interpolating function

        # apply transformation to input image
        return torch.from_numpy(f(input_image))
    
    def hist_normalize(self, images):

        if isinstance(images, list):

            new_images = []
            for img in images:
                new_img = torch.zeros_like(img)
                for j in range(len(img)):
                    new_img[j, ...] = self._do_hist_normalization_img(img[j, ...])
            new_images.append(new_img)
        
        elif isinstance(images, torch.Tensor):
            new_images = torch.zeros_like(images)
            for j in range(len(images)):
                new_images[j, ...] = self._do_hist_normalization_img(images[j, ...])

        return new_images

    def nyul_train_and_normalize(self, images):

        self.nyul_train_standard_scale(images)
        return self.hist_normalize(images)

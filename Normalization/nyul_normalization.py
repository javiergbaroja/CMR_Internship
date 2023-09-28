import nibabel as nib
from scipy.interpolate import interp1d
import numpy as np


def get_landmarks(img, percs):
    landmarks = np.percentile(img, percs)
    return landmarks


def nyul_train_standard_scale(all_imgs,
                              i_min=0,
                              i_max=100,
                              i_s_min=0,
                              i_s_max=1,
                              l_percentile=10,
                              u_percentile=90,
                              step=50):

    percs = np.concatenate(([i_min],
                            np.arange(l_percentile, u_percentile+1, step),
                            [i_max]))

    standard_scale = np.zeros(len(percs))

    # process each image in order to build the standard scale
    for i, img in enumerate(all_imgs):
        landmarks = get_landmarks(img, percs)
        min_p = np.percentile(img, i_min)
        max_p = np.percentile(img, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])  # create interpolating function
        landmarks = np.array(f(landmarks)) # interpolate landmarks
        standard_scale += landmarks  # add landmark values of this volume to standard_scale

    standard_scale = standard_scale / len(all_imgs)  # get mean values
    return standard_scale, percs



def do_hist_normalization(input_image,
                          landmark_percs,
                          standard_scale,
                          mask=None,
                          interp_type='linear'):

    landmarks = get_landmarks(input_image, landmark_percs)
    f = interp1d(landmarks, standard_scale, kind=interp_type, fill_value='extrapolate')  # define interpolating function

    # apply transformation to input image
    return f(input_image)
import numpy as np
from skimage.transform import warp_polar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


from .rotation_plot import get_center


def get_ratio_for_sector(mask_polar, angle_start, angle_end):
    sector = np.round(mask_polar[angle_start:angle_end])
    myo = np.count_nonzero(sector > 0)
    sd2 = np.count_nonzero(sector > 1)
    sd5 = np.count_nonzero(sector > 2)
    sd2_ratio = sd2/myo
    sd5_ratio = sd5/myo
    return sd2_ratio, sd5_ratio


def initialize_sectors(mask):
    n_slices = mask.shape[0]
    sectors = np.zeros((n_slices, 6))
    return sectors


def fill_sectors(mask):
    sd2_sectors = initialize_sectors(mask)
    sd5_sectors = initialize_sectors(mask)
    for s, mask_slice in enumerate(mask):
        center = get_center(np.where(mask_slice > 0, 1, 0).astype('float64'))
        mask_slice_polar = warp_polar(mask_slice, center=(center[1],center[0]), radius=128 - max(center), preserve_range=True)
        for angle in range(6):
            sd2_ratio, sd5_ratio = get_ratio_for_sector(mask_slice_polar, 60*angle, 60*angle + 60)
            sd2_sectors[s][angle] = sd2_ratio
            sd5_sectors[s][angle] = sd5_ratio
    return sd2_sectors, sd5_sectors


def bullseye_plot(data, cmap=None, norm=None, axs=None):
    linewidth = 2

    fig = plt.figure(figsize=(10, 5), layout="constrained")
    fig.get_layout_engine().set(wspace=.1, w_pad=.2)
    if axs == None:
        axs = fig.subplots(1, 1, subplot_kw=dict(projection='polar'))
        axs.grid(False)
        
    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2 * np.pi, 768)
    r = np.linspace(0.2, 1, data.shape[0]+1)

    # Create the segments
    for i in range(r.shape[0]):
        axs.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)
    
    for i in range(6):
        theta_i = np.deg2rad(i * 60)
        axs.plot([theta_i, theta_i], [r[0], 1], '-k', lw=linewidth)
    
    # Fill the segments
    for n_slice in range(data.shape[0]):
        r0 = r[n_slice:(n_slice+2)]
        r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T

        for i in range(6):
            theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(360)
            theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
            z = np.ones((128 - 1, 2 - 1)) * data[n_slice][5-i]
            axs.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm, shading='auto')
        
    axs.set_ylim([0, 1])
    axs.set_yticklabels([])
    axs.set_xticklabels([])
    
    norm = colors.Normalize(0, 1)

    fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm),
                 cax=axs.inset_axes([0, -.15, 1, .1]),
                 orientation='horizontal', label='scar density')








# import numpy as np
# from skimage.transform import warp_polar
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# from .rotation_plot import get_center


# def get_dice_scores_for_sector(sector_pred, sector_gt):
#     sd2_fwhm_pred_bin = np.where(sector_pred > 1, 1, 0)
#     sd2_fwhm_gt_bin = np.where(sector_gt > 1, 1, 0)
#     sd2_fwhm_dice = np.sum(sd2_fwhm_pred_bin[sd2_fwhm_gt_bin==1])*2.0 / (np.sum(sd2_fwhm_pred_bin) + np.sum(sd2_fwhm_gt_bin))

#     sd5_pred_bin = np.where(sector_pred > 2, 1, 0)
#     sd5_gt_bin = np.where(sector_gt > 2, 1, 0)
#     sd5_dice = np.sum(sd5_pred_bin[sd5_gt_bin==1])*2.0 / (np.sum(sd5_pred_bin) + np.sum(sd5_gt_bin))
#     return sd2_fwhm_dice, sd5_dice


# def get_ratio_for_sector(mask_polar, angle_start, angle_end, mask_polar_gt=None):
#     sector = np.round(mask_polar[angle_start:angle_end])
    
#     myo = np.count_nonzero(sector > 0)
#     sd2_fwhm = np.count_nonzero(sector > 1)
#     sd5 = np.count_nonzero(sector > 2)
#     sd2_fwhm_ratio = sd2_fwhm/myo
#     sd5_ratio = sd5/myo
    
#     if isinstance(mask_polar, np.ndarray):
#         sector_gt = np.round(mask_polar_gt[angle_start:angle_end])
#         sd2_fwhm_dice, sd5_dice = get_dice_scores_for_sector(sector, sector_gt)
#         return sd2_fwhm_ratio, sd5_ratio, sd2_fwhm_dice, sd5_dice
    
#     return sd2_fwhm_ratio, sd5_ratio, None, None


# def initialize_sectors(mask):
#     n_slices = mask.shape[0]
#     sectors = np.zeros((n_slices, 6))
#     return sectors


# def fill_sectors(mask, mask_gt=None):
#     mask_polar_gt = mask_gt
#     sd2_sectors = initialize_sectors(mask)
#     sd5_sectors = initialize_sectors(mask)
#     sd2_sectors_dice = initialize_sectors(mask)
#     sd5_sectors_dice = initialize_sectors(mask)
#     for s, mask_slice in enumerate(mask):
#         center = get_center(np.where(mask_slice > 0, 1, 0).astype('float64'))
#         mask_slice_polar = warp_polar(mask_slice, center=(center[1],center[0]), radius=128 - max(center), preserve_range=True)
#         if isinstance(mask_gt, np.ndarray):
#             mask_polar_gt = warp_polar(mask_gt[s], center=(center[1],center[0]), radius=128 - max(center), preserve_range=True) 
#         for angle in range(6):
#             sd2_fwhm_ratio, sd5_ratio, sd2_fwhm_dice, sd5_dice = get_ratio_for_sector(mask_slice_polar, 60*angle, 60*angle + 60, mask_polar_gt)
#             sd2_sectors[s][angle] = sd2_fwhm_ratio
#             sd5_sectors[s][angle] = sd5_ratio
#             sd2_sectors_dice[s][angle] = sd2_fwhm_dice
#             sd5_sectors_dice[s][angle] = sd5_dice
#     return sd2_sectors, sd5_sectors, sd2_sectors_dice, sd5_sectors_dice


# def bullseye_plot(data, cmap=None, norm=None, dice_scores=None):
#     linewidth = 2

#     fig = plt.figure(figsize=(10, 5), layout="constrained")
#     fig.get_layout_engine().set(wspace=.1, w_pad=.2)
#     axs = fig.subplots(1, 1, subplot_kw=dict(projection='polar'))
#     axs.grid(False)
        
#     if cmap is None:
#         cmap = plt.cm.viridis

#     if norm is None:
#         norm = colors.Normalize(vmin=data.min(), vmax=data.max())

#     theta = np.linspace(0, 2 * np.pi, 768)
#     r = np.linspace(0.2, 1, data.shape[0]+1)

#     # Create the segments
#     for i in range(r.shape[0]):
#         axs.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)
    
#     for i in range(6):
#         theta_i = np.deg2rad(i * 60)
#         theta_i_next = np.deg2rad(((i+1) % 7) * 60)
#         axs.plot([theta_i, theta_i], [r[0], 1], '-k', lw=linewidth)
#         if isinstance(dice_scores, np.ndarray):
#             for j in range(data.shape[0]):
#                 axs.text((theta_i + theta_i_next)/2, r[j]+0.06, np.round(dice_scores[j][i], 2), horizontalalignment='center',
#                         fontsize='small', color='darkgreen')
    
#     # Fill the segments
#     for n_slice in range(data.shape[0]):
#         r0 = r[n_slice:(n_slice+2)]
#         r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T

#         for i in range(6):
#             theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(360)
#             theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
#             z = np.ones((128 - 1, 2 - 1)) * data[n_slice][5-i]
#             axs.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm, shading='auto')
        
#     axs.set_ylim([0, 1])
#     axs.set_yticklabels([])
#     axs.set_xticklabels([])
    
#     norm = colors.Normalize(0, 1)

#     fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm),
#                  cax=axs.inset_axes([0, -.15, 1, .1]),
#                  orientation='horizontal', label='scar density')
import cv2
from scipy.spatial import distance
from bresenham import bresenham
import scipy.interpolate as interp
import numpy as np
import time


def get_center(mask):
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def get_min_endo_max_epi(mask, center):
    min_endo = mask.shape[1]
    max_epi = 0
    mask_coords = np.argwhere(mask==1)
    for c in mask_coords:
        dist = distance.euclidean((c[1], c[0]), center)
        if dist < min_endo:
            min_endo = dist
            endo_coord = c
        if dist > max_epi:
            max_epi = dist
            epi_coord = c
    return np.max((np.floor(min_endo)-1, 0)), np.min((mask.shape[1], np.ceil(max_epi)+1)), endo_coord, epi_coord


def get_endo_epi_points(img, center, r_min, r_max, n_samples):
    angles = np.linspace(0, 2*np.pi, num=n_samples)
    endo_points = []
    epi_points = []
    for t in angles:
        x = center[0] + np.round(r_min * np.cos(t))
        y = center[1] + np.round(r_min * np.sin(t))
        if (x >= 0) & (x <= img.shape[0]-1) & (y >= 0) & (y <= img.shape[1]-1):
            endo_points.append((x,y))
        
        x = center[0] + np.round(r_max * np.cos(t))
        y = center[1] + np.round(r_max * np.sin(t))
        if (x >= 0) & (x <= img.shape[0]-1) & (y >= 0) & (y <= img.shape[1]-1):
            epi_points.append((x,y))
    return endo_points, epi_points


def get_intensity_line(img, p1, p2):
    # bresenham function finds pixels along a line
    coords = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    x, y = np.transpose(np.array(coords))
    pixel_intensities = img[y,x]
    return pixel_intensities


def equal_size(intensities):
    # get smallest length
    length = intensities[0].shape[0]
    for i in intensities:
        if i.shape[0] < length:
            length = i.shape[0]
        
    pixel_intensities = []
    for i in intensities:
        interpolate = interp.interp1d(np.arange(i.size),i)
        new_size = interpolate(np.linspace(0,i.size-1, length))
        pixel_intensities.append(new_size)
    return np.array(pixel_intensities)


def straighten_endo(arr, img=None):
    normed_intensity = []
    if img is not None:
        normed_img = []
    for i, col in enumerate(arr):
        n_zeros = 0
        for pix in col:
            if pix == 0:
                n_zeros += 1
            else:
                break
        normed_intensity.append(np.concatenate((col[n_zeros::], np.zeros(n_zeros))))

        if img is not None:
            normed_img.append(np.concatenate((img[i][n_zeros::], np.zeros(n_zeros))))
    if img is not None:
        return np.array(normed_intensity), np.array(normed_img)
    return np.array(normed_intensity)


def straighten_epi(arr, img=None):
    normed_intensity = []
    if img is not None:
        normed_img = []
    for i, col in enumerate(arr):
        n_zeros = 0
        for pix in col:
            if pix == 0:
                n_zeros += 1
            else:
                break
        normed_intensity.append(np.concatenate((col[n_zeros::], np.zeros(n_zeros))))

        if img is not None:
            normed_img.append(np.concatenate((img[i][n_zeros::], np.zeros(n_zeros))))
    if img is not None:
        return np.array(normed_intensity), np.array(normed_img)
    return np.array(normed_intensity)


def get_edge(bin_mask):
    edge_mask = np.zeros(bin_mask.shape)
    for i in range(bin_mask.shape[0]):
        for j in range(bin_mask.shape[1]):
            if bin_mask[i][j] == 1:
                if i+1 < bin_mask.shape[0]:
                    if bin_mask[i+1][j] != 1:
                        edge_mask[i][j] = 1
                if j+1 < bin_mask.shape[1]:
                    if bin_mask[i][j+1] != 1:
                        edge_mask[i][j] = 1
                if i-1 >= 0:
                    if bin_mask[i-1][j] != 1:
                        edge_mask[i][j] = 1
                if j-1 >= 0:
                    if bin_mask[i][j-1] != 1:
                        edge_mask[i][j] = 1
    return edge_mask


def circular_plot(img, mask, pred, endo_points, epi_points):
    
    # get new pixel intensities
    pixel_intensities_img = []
    pixel_intensities_mask = []
    pixel_intensities_pred = []
    for j in range(len(endo_points)):
        pixel_intensities_img.append(get_intensity_line(img, endo_points[j], epi_points[j]))
        circ_img = equal_size(pixel_intensities_img)

        pixel_intensities_mask.append(get_intensity_line(mask, endo_points[j], epi_points[j]))
        circ_mask = np.round(equal_size(pixel_intensities_mask))

        pixel_intensities_pred.append(get_intensity_line(pred, endo_points[j], epi_points[j]))
        pred_circ = np.round(equal_size(pixel_intensities_pred))
    
    # straighten the endocardium
    normed_mask, normed_img = straighten_endo(circ_mask, img=circ_img)
    pred_normed_mask = straighten_endo(pred_circ)
    # get edge
    pred_edge_myo = get_edge(np.where(pred_normed_mask >= 1, 1, 0))
    pred_edge_scar = get_edge(np.where(pred_normed_mask == 2, 1, 0))
    return normed_img, normed_mask, pred_edge_myo, pred_edge_scar
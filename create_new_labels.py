from DataLoader.cohorts import all_cohorts
import os
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy import ndimage
import matplotlib.pyplot as plt
from Helpers.data_utils import adjust_order


def filter_isolated_cells(arr, struct=np.ones((3,3))):
    '''
    Removes single (isolated) pixels from the given mask (arr).
    struct defines the neighbourhood, (i.e what counts as a neighbour).
    Taken from from stackoverflow.com/questions/28274091
    '''
    id_regions, num_ids = ndimage.label(arr, structure=struct)
    id_sizes = np.array(ndimage.sum(arr, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    arr[area_mask[id_regions]] = 0
    return arr


def makeLGELabels(images, contours):
    
    myo = (np.sum(contours[...,6:7], axis=-1)>0)*1
    healthy = (np.sum(contours[...,3:5], axis=-1)>0)*1
    artifact = (np.sum(contours[...,5:6], axis=-1)>0)*1
    not_artifact = 1-artifact

    # get center slice
    n_middle = round(images.shape[0]/2)
    myo_healthy_vals = images[n_middle][healthy[n_middle]>0]
    myo_all_vals = images[n_middle][myo[n_middle]>0]
    
    i = 1
    if (len(myo_healthy_vals) == 0) and (len(myo_all_vals) > 0):
        print('healthy region missing')
        # try adjacent slices
        while ((n_middle + i) < images.shape[0]) and (len(myo_healthy_vals) == 0) and (len(myo_all_vals) > 0):
            n_middle += i
            myo_healthy_vals = images[n_middle][healthy[n_middle]>0]
            myo_all_vals = images[n_middle][myo[n_middle]>0]
            if (len(myo_healthy_vals) > 0) and (len(myo_all_vals) > 0):
                print('found healthy region in adject slice')


    for i in range(images.shape[0]): # loop through each slice

        if len(myo_healthy_vals) > 0:
            m, sd = np.mean(myo_healthy_vals), np.std(myo_healthy_vals)
            sd2 = ((np.clip(images[i]-(m+2*sd), 0, np.inf) * myo[i]) > 0)*1
            sd5 = ((np.clip(images[i]-(m+5*sd), 0, np.inf) * myo[i]) > 0)*1

            #remove isolated pixels:
            sd2 = filter_isolated_cells(sd2)
            sd5 *= sd2

        else:
            sd2 = np.zeros(myo[i].shape, dtype='int')
            sd5 = np.zeros(myo[i].shape, dtype='int')



        shunk_myo_i = binary_erosion(myo[i], iterations=2)*1

        if np.sum(shunk_myo_i)>0:
            # mean_val = np.mean(myo_all_vals)

            # im_from_0 = im[i].astype('float') - np.min(myo_all_vals)#np.percentile(myo_all_vals, 0.01) #slightly more noise resistant than just using the min
            # im_from_0 = im[i].astype('float') - np.percentile(im[i], 0.05) #slightly more noise resistant than just using the min

            # print(myo[i].max(), myo[i].min(), myo[i].shape)

            # imsave('shrinking_test.png', (shunk_myo_i+myo[i])*128)
            # print(shunk_myo_i.max(), shunk_myo_i.min(), shunk_myo_i.shape)
            # myo_all_vals_from_0 = im_from_0[myo[i]>0]

            myo_all_vals_from_0 = images[i][shunk_myo_i>0] #values in the shrunken region
            # myo_min_min = np.min(myo_all_vals_from_0)
            myo_min = np.percentile(myo_all_vals_from_0, 0.5)
            # print(myo_min, myo_min_min)

            im_from_0 = images[i].astype('float') - myo_min #subtract the min of the shrunken region
            temptemp = im_from_0 - (np.max(myo_all_vals_from_0-myo_min)/2.) #FWHM with the max of the shrunken region

            # print('---->')
            # print(im_from_0.min(), im_from_0.max(), np.max(myo_all_vals_from_0)/2. )
            # print( temptemp.min(), temptemp.max() )

            # fwhm = ((np.clip(im[i]*1.-np.max(myo_all_vals)/2., 0, np.inf) * myo[i]) > 0)*1.0

            fwhm = (np.clip(temptemp, 0, np.inf) > 0) * myo[i]
            fwhm = filter_isolated_cells(fwhm)

            # print(np.unique(fwhm))
            # print( 'np.sum(fwhm)', np.sum(fwhm), np.sum(myo[i]) )

        else:
            fwhm = np.zeros(myo[i].shape, dtype='int')

        sd2 = sd2 * not_artifact[i]
        sd5 = sd5 * not_artifact[i]

        fwhm = fwhm  * not_artifact[i]

        lge_slice = np.concatenate([np.expand_dims(myo[i],-1), np.expand_dims(sd2,-1), np.expand_dims(sd5,-1), np.expand_dims(fwhm,-1)], -1)
        contours[i, :, :, 6:10] = lge_slice

    return contours



for cohort in all_cohorts:
    if not os.path.exists(os.path.join('DataNew', cohort)):
        os.makedirs(os.path.join('DataNew', cohort), exist_ok=True)
    for patient in os.listdir(os.path.join('Data', cohort)):
        if patient == "p203" and cohort == "pisa": 
            print("here")
        if patient.endswith(".csv"): continue
        if not os.path.exists(os.path.join(os.path.join('DataNew', cohort), patient)):
            os.makedirs(os.path.join(os.path.join('DataNew', cohort), patient), exist_ok=True)

        img = np.load(os.path.join('Data', cohort, patient, 'images.npy'))
        mask = np.load(os.path.join('Data', cohort, patient, 'labels.npy'))
        positions = np.load(os.path.join('Data', cohort, patient, 'positions.npy'))
        qualities = np.load(os.path.join('Data', cohort, patient, 'qualities.npy'))
        slice_thickness = np.load(os.path.join('Data', cohort, patient, 'slice_thickness.npy'))

        to_keep = np.sum(mask[...,6:], (1,2,3)) != 0
        img = img[to_keep]
        mask = mask[to_keep]
        assert len(img) == len(mask)
        if len(positions) > len(to_keep):
            print(f"Patient {cohort}_{patient} more positions than slices")
            positions = positions[:len(to_keep)][to_keep]
        elif len(positions) < len(to_keep):
            print(f"Patient {cohort}_{patient} less positions than slices")
            positions = positions[to_keep[:len(positions)]]

        qualities = qualities[to_keep]
        slice_thickness = slice_thickness[to_keep]

        indeces = np.array([i for i in range(len(img))])
        indeces = adjust_order(positions, indeces)
        img = img[indeces]
        mask = mask[indeces]
        positions = positions[indeces]
        qualities = qualities[indeces]

        # get new contours
        contours = makeLGELabels(img, mask)

        # save everything into NewData
        np.save(os.path.join('DataNew', cohort, patient, 'images.npy'), img)
        np.save(os.path.join('DataNew', cohort, patient, 'labels.npy'), contours)
        np.save(os.path.join('DataNew', cohort, patient, 'positions.npy'), positions)
        np.save(os.path.join('DataNew', cohort, patient, 'qualities.npy'), qualities)
        np.save(os.path.join('DataNew', cohort, patient, 'slice_thickness.npy'), slice_thickness)
        
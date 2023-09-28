import numpy as np
import torchvision.transforms as T
import copy

from .mydataset import ImageDataset
from Helpers.utils import shuffle_lists

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

def get_ImageDataset3D(data:dict, even_dataset:bool, augment:bool=False, batch_size:int=-1):
   
        if even_dataset:
            return byPatientDataset3D(data=data, augment=augment)
        else:
            return bySliceImageDataset3D(data=data, batch_size=batch_size, augment=augment)



class byPatientDataset3D(ImageDataset): # Same as ImageDataset
    """This Dataset will take in all the patient images and group them in batches. 
    The batches will have the same number of patients. Better if used for cases when there is a regular number of slices for each patient.
    """
    def __init__(self,
                 data:dict,
                 augment:bool=False) -> None:
        
        self.images = data["images"]
        self.masks = data["masks"]
        self.pIDs = data["pIDs"]
        self.augment = augment

        self.n = len(self.images)
    
    def __getitem__(self, index):

        img = self.images[index % self.n] + 0.
        msk = self.masks[index % self.n] + 0
        pID = self.pIDs[index % self.n]

        if self.augment:
            img, msk = self.transforms(img, msk)
        # batch['image'].shape = (Batch, Seq, Channel, Height, Width)
        return {'image': img, 'mask': msk, 'pID': pID}
    
    def __repr__(self):
        rep = 'ImageDataset3D(even_dataset=True' + ', augment=' + str(self.augment) + ')'
        return rep
    

class bySliceImageDataset3D(ImageDataset):
    """This Dataset will take in all the patient images and group them in batches. 
    It groups as patients into the same batch to have a number of slices as close to the selected batch_size as possible.
        The DataLoader should:
            always have a batch_size=1
            always have shuffle=False
    """

    def __init__(self, 
                 data:dict,
                 batch_size:int, 
                 augment:bool=False):

        self.augment = augment
        self.batch_size = batch_size
        self.images = data["images"]
        self.pIDs = data["pIDs"]
        self.masks = data["masks"]

        if batch_size != -1 and batch_size > min([vol.shape[0] for vol in self.images]):
            self._group_by_batch()
        else:
            self.batch_pos = [[i] for i in range(len(self.images))]
            self.n_batches = len(self.images)
        
        self.n_patients = len(self.images)
        
        # self.mean_slices_per_patient = np.mean([image.shape[0] for image in images])

        self.extracted_batches = 0

    def _group_by_batch(self):
        # Step 1: shuffle images
        self.images, self.masks, self.pIDs = shuffle_lists(self.images, self.masks, self.pIDs)

        # Step 2: create list containing list_index and # slices of a patient
        self.n_slices_per_patient = [[i, self.images[i].shape[0]] for i in range(len(self.images))]
        
        # Step 3: Initializations
        grouped_pos = [[self.n_slices_per_patient[0][0]]]
        # grouped_slice_n = [[self.n_slices_per_patient[0][1]]]
        count = self.n_slices_per_patient[0][1]
        n_slices_per_patient = copy.deepcopy(self.n_slices_per_patient)

        i = 1
        loopcount = len(n_slices_per_patient)

        # Loop through positions left in n_slices_per_patient (it is updated in the loop)
        while i < loopcount:
            added = False
            # if patient already part of current batch does not provide enough slices for a batch
            # a for loop tries to find another patient without surpassing the batch_size
            # If the patient is found, it is removed from the list to avoid duplicates
            if count <= self.batch_size: 
                for j in range(i+1, len(n_slices_per_patient)):
                    if count + n_slices_per_patient[j][1] <= self.batch_size:
                        grouped_pos[-1].append(n_slices_per_patient[j][0])
                        # grouped_slice_n[-1].append(n_slices_per_patient[j][1])

                        count += n_slices_per_patient[j][1]
                        added = True
                        del n_slices_per_patient[j]
                        loopcount = len(n_slices_per_patient)
                        break

                if not added:
                    count = n_slices_per_patient[i][1]
                    grouped_pos.append([n_slices_per_patient[i][0]])
                    # grouped_slice_n.append([n_slices_per_patient[i][1]])
            else:
                count = n_slices_per_patient[i][1]
                grouped_pos.append(n_slices_per_patient[i][0])
                # grouped_slice_n.append([n_slices_per_patient[i][1]])
            if not added:
                i+=1 

        self.batch_pos = grouped_pos
        self.n_batches = len(grouped_pos)   

    
    def __getitem__(self, index) -> dict:

        self.extracted_batches += 1
        image_batch = []
        msk_batch = []
        pIDs_batch = []

        for i in self.batch_pos[index]:
            if self.augment:
                img, msk = self.transforms(self.images[i], self.masks[i])
                image_batch.append(img)
                msk_batch.append(msk)
            else:
                image_batch.append(self.images[i])
                msk_batch.append(self.masks[i])
            pIDs_batch.append(self.pIDs[i])

        assert len(msk_batch) == len(image_batch)
        
        # If finished iterating through dataset, reshufle
        if self.extracted_batches >= self.n_batches and self.augment:
            self._group_by_batch()

        return {'image': image_batch, 'mask': msk_batch, 'pID': pIDs_batch}


    def __len__(self):

        return self.n_batches
    
    def __repr__(self):
        rep = 'ImageDataset3D(even_dataset=False, batch_size=' + str(self.batch_size) + ', augment=' + str(self.augment) + ')'
        return rep
    

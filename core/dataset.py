from torch.utils.data import Dataset
import cv2
import numpy as np
import torch    
import glob
import os
import re
import random
from sklearn.model_selection import train_test_split


import config

# -----------------------------------------------------------------------------
# Custom dataset class inheriting from pytorch Dataset
# -----------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    
    def __init__(self, pth_data, transform = None, train = True, test_size = .2, seed = 42):
        
        # inherit paret methods
        super().__init__()
        
        # save images and corresponding masks
        self.pth_data = pth_data
        self.img_mask_dict = None
        self.transform = transform

        # initilize the img_mask_dict
        self.initialize_img_mask_dict()

        # train / test
        items = list(self.img_mask_dict.items())
        
        # Split the list into train and test sets
        train_items, test_items = train_test_split(items, test_size=test_size, random_state=seed)
        
        # Convert the train and test sets back into dictionaries
        train_dict = dict(train_items)
        test_dict = dict(test_items)
        if train:
            self.img_mask_dict = train_dict
        else: 
            self.img_mask_dict = test_dict
    

    def initialize_img_mask_dict(self):
        
        def get_type_and_id(pth_png):
            type = ''
            id = None
            
            png_name = os.path.basename(pth_png)
        
            if 'malignant' in png_name:
                type = 'malignant'
            if 'benign' in png_name:
                type = 'benign'
            if 'normal' in png_name:
                type = 'normal'  
        
            id = re.search(r'\((\d+)\)', png_name).group(1)
        
            return type, id
    
        def get_masks(pth_image, pth_masks):
            
            img_type, img_id = get_type_and_id(pth_image)
        
            masks = []
        
            for pth_mask in pth_masks:
                mask_type, mask_id = get_type_and_id(pth_mask)
                if mask_type == img_type and mask_id == img_id:
                    masks.append(pth_mask)
            return masks
    
        # paths of all pngs
        pth_pngs = glob.glob(os.path.join(self.pth_data,'**','*.png'))
        
        # path of images
        # pth_images = [pth for pth in pth_pngs if 'mask' not in pth]
        pth_images = [pth for pth in pth_pngs if 'mask' not in pth and 'normal' not in pth]

        
        # path of masks
        pth_masks = [pth for pth in pth_pngs if 'mask' in pth]
        
        # image masks paths list dict
        self.img_mask_dict = {pth_image : get_masks(pth_image, pth_masks) for pth_image in pth_images}
        
    def __len__(self): 
        return len(self.img_mask_dict)
    
    def __getitem__(self, index):
        
        def merge_masks(masks):
            sum_maks = None
            for idx,mask in enumerate(masks):
                if idx == 0:
                    sum_maks = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
                else:
                    sum_maks += cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
            return sum_maks / 255

        # read img and masks paths
        pth_img, pth_masks = list(self.img_mask_dict.items())[index]

        # convert to numpy the image
        image = cv2.imread(pth_img,cv2.IMREAD_GRAYSCALE)

        # convert to numpy the mask
        mask = merge_masks(pth_masks)
        
        # apply transformations if needed
        if self.transform is not None:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            
        image = image.float()
        mask = mask.float()

        return image, mask
















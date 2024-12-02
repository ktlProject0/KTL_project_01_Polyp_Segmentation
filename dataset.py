import os
import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from preprocess import get_image,get_transform
from torchvision.datasets.folder import default_loader

class CustomDataset(Dataset):
    def __init__(self,direc,mode='eval'):
        self.mode = mode
        img_path = natsorted(glob.glob(os.path.join(direc,'images','*')))
        mask_path = natsorted(glob.glob(os.path.join(direc,'masks','*')))
        self.meta_df = pd.DataFrame({"image":img_path,'label':mask_path})
        self.transform =get_transform(self.mode)
        self.cache={}
        
    def __len__(self):
        return len(self.meta_df)

    
    def __getitem__(self,idx):
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            sample = self.meta_df.iloc[idx,:].to_dict()
            image = get_image(sample['image']).astype(np.float32)
            
            mask = np.array(default_loader(sample['label']))[...,0]
            mask = mask>100
            mask = np.stack([(mask==x).astype(np.uint8) for x in [0,1]], axis=-1)

            sample['image'] = image
            sample['mask'] = mask
            sample['origin_shape'] = image.shape

            self.cache[idx] = sample
            
        if self.transform:
            transformed = self.transform(image= sample['image'], mask = sample['mask'])

        sample_input = {}
        sample_input['input'] = transformed['image']
        sample_input['target'] = transformed['mask']
        sample_input['origin_shape'] = sample['origin_shape']
        
        return sample_input
        
if __name__ == '__main__':
    train = CustomDataset('./data/train/')
    test = CustomDataset('./data/test/')
    for sample_input in train:
        print(sample_input['input'].shape)
        print(sample_input['target'].shape)
        break
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

class SemanticDataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 split='train',
                 crop_size=64
                ):
        self.X  = X
        self.y = y
        self.split = split
        self.crop_size = crop_size   
                
    def __len__(self):
        return self.X.shape[0]
    
    def augment_data(self, raw, mask):
        
        transform = A.Compose([
              A.RandomCrop(width=self.crop_size, height=self.crop_size),
              A.HorizontalFlip(p=0.5),
              A.VerticalFlip(p=0.5)
            ])

        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']
       
        return raw, mask
       
    def __getitem__(self, idx):
        
        raw = self.X[idx,:, :] # load raw to numpy array
        mask = self.y[idx, :, :] # load mask to numpy array
        
        # if training, run augmentations
        #if self.split == 'train':
        raw, mask = self.augment_data(raw, mask)
       
        #mask = (fg != 0).astype(np.float32)

        # add channel dim for network
        if len(raw.shape) == 2:
            raw = np.expand_dims(raw, axis=0)
        elif len(raw.shape) == 3:
            raw = np.transpose(raw, axes=(-1, *range(raw.ndim - 1)))
        
        mask = np.expand_dims(mask, axis=0)
        raw = raw.astype(np.float32)

        return raw, mask
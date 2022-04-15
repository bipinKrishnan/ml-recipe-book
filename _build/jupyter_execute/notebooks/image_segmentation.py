#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q segmentation_models_pytorch')


# In[ ]:


from pathlib import Path
import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer


# In[ ]:


img_root_path = Path("../input/kittiroadsegmentation/training/image_2")
mask_root_path = Path("../input/kittiroadsegmentation/training/gt_image_2")
img_files = img_root_path.glob('*')

def get_existing_imgs_and_masks(img_files, mask_root_path):
    existing_imgs, existing_masks = [], []
    
    for img_file in img_files:
        mask_file = f"{mask_root_path}/um_lane_{str(img_file).split('_')[-1]}"

        if os.path.exists(mask_file):
            existing_imgs.append(img_file)
            existing_masks.append(mask_file) 
         
    return existing_imgs, existing_masks


# In[ ]:


imgs, masks = get_existing_imgs_and_masks(img_files, mask_root_path)


# In[ ]:


df = pd.DataFrame(columns=['imgs'])

df['imgs'] = imgs
df['masks'] = masks


# In[ ]:


train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)


# In[ ]:


sample_img_path = str(imgs[0])
sample_mask_path = str(masks[0])


# In[ ]:


sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)/255.0


# In[ ]:


sample_mask = cv2.imread(sample_mask_path)/255.0
sample_mask = sample_mask[:, :, 0]
sample_mask = (sample_mask==1).astype(float)


# In[ ]:


# resize and convert to tensors
transform = A.Compose([A.Resize(256, 256), ToTensorV2()])

augmented = transform(image=sample_img, mask=sample_mask)


# In[ ]:


augmented['mask'] = augmented['mask'].unsqueeze(0)
augmented['image'] = augmented['image'].type(torch.FloatTensor)


# In[ ]:


class LoadDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.imgs = img_paths
        self.masks = mask_paths
        self.transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
        
    def __len__(self): return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        
        mask = cv2.imread(str(self.masks[idx]))/255.0
        mask = (mask[:, :, 0]==1).astype(float)
        
        augmented = self.transform(image=img, mask=mask)
        augmented['image'] = augmented['image'].type(torch.FloatTensor)
        augmented['mask'] = augmented['mask'].unsqueeze(0)
        
        return augmented


# In[ ]:


train_ds = LoadDataset(train_df['imgs'].values, train_df['masks'].values)
eval_ds = LoadDataset(eval_df['imgs'].values, eval_df['masks'].values)


# In[ ]:


print(train_ds[0], eval_ds[0])


# In[ ]:


class SegmentationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.batch_size = 32
        
        self.model = smp.Unet(
            'resnet34', 
            classes=1, 
            activation=None, 
            encoder_weights='imagenet'
        )
        
    def forward(self, x):
        return self.model(x)
        
    def train_dataloader(self):
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)


# In[ ]:


model = SegmentationModel()
trainer = Trainer(
    accelerator='auto',  # automatically select the available accelerator(CPU, GPU, TPU etc)
    devices=1,           # select the available one device of the accelerator
    auto_lr_find=True,   # use learning rate finder to set the learning rate
    max_epochs=40,        # number of epochs to train
)

trainer.tune(model)      # runs learning rate finder and sets the learning rate
trainer.fit(model)


# In[ ]:


idx = 24
img = eval_ds[idx]['image']
mask = eval_ds[idx]['mask']

# prediction
model.eval()
with torch.no_grad():
    pred_probability = model(img.unsqueeze(0))
    pred_probability = torch.sigmoid(pred_probability)
    pred = (pred_probability>0.5).type(torch.int)


# In[ ]:


plt.subplot(1, 2, 1)
plt.imshow(pred.squeeze())
plt.title('Prediction')

plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze())
plt.title("Original");


# In[ ]:





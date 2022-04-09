#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -q timm


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PIL import Image
from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
import timm
from pytorch_lightning.loggers import WandbLogger


# In[3]:


csv_path = "../input/animal-images-dataset/animal_data_img.csv"
df = pd.read_csv(
    csv_path,
    usecols=['Animal_Type', 'Image_File']
    )
df.head()


# In[4]:


print(df.head())


# In[5]:


# remove rows with 'Guinea pig / mouse' and 'Other' labels
df = df.query("Animal_Type not in ['Guinea pig / mouse', 'Other']").reset_index(drop=True)


# In[6]:


label_string = df['Animal_Type'].unique()
label_int = range(len(label_string))

# create a dictionary with string to int label mapping
label_mapping = dict(zip(label_string, label_int))
print(label_mapping)


# In[7]:


df['labels'] = df['Animal_Type'].map(label_mapping)


# In[8]:


class LoadDataset(Dataset):
    def __init__(self, df):
        self.root_dir = Path("../input/animal-images-dataset/animal_images")
        # all the image paths are stores here
        self.images = df['Image_File'].values
        # all the labels are stored here
        self.labels = df['labels'].values
        
        # these transforms are applied to each image
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop((100, 100)),
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img_path = self.root_dir/self.images[idx]
        # load the image and pply the transforms
        image = Image.open(img_path)
        image = self.transforms(image)
        # load the label corresponding to the above image
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (image, label)
    
    def __len__(self): return len(self.images)


# In[9]:


train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    shuffle=True, 
    stratify=df['labels'], 
    random_state=42,
    )


# In[10]:


# training set
train_ds = LoadDataset(train_df)
# test set
test_ds = LoadDataset(test_df)


# In[ ]:


train_ds[0]


# In[11]:


class AnimalModel(LightningModule):
    def __init__(self):
        super().__init__()
        # hyper-parameters for training the model
        self.batch_size = 64
        self.learning_rate = 1e-7

        # create a pretrained resnet34 by specifying the number of labels to classify
        self.model = timm.create_model(
            "resnet34", 
            pretrained=True, 
            num_classes=len(label_int)
        )

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
    
    # return validation/evaluation dataloader
    def val_dataloader(self):
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    # return the optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        
        # this is how we log stuff and show it along with the progress bar(prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.learning_rate)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        
        self.log("eval_loss", loss, prog_bar=True)
        return loss


# In[22]:


logger = WandbLogger(project='lightning-project', name='animal-clf-test', log_model=True)

trainer = Trainer(
    accelerator='auto', 
    auto_lr_find=True,  
    max_epochs=10,      
    devices=1,
    logger=logger, # wandb logger
)


# In[23]:


model = AnimalModel()
trainer.tune(model)


# In[24]:


trainer.fit(model)


# In[40]:


import matplotlib.pyplot as plt
sample = test_ds[19]

pred = model(sample[0].unsqueeze(0))
pred = torch.argmax(pred, dim=1).item()

print(label_mapping)
# final prediction
print(f"Predicted class: {pred}")

plt.imshow(torch.permute(sample[0], (1, 2, 0)));


# In[ ]:





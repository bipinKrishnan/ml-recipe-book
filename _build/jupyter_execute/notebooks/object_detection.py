#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_lightning import LightningModule, Trainer

import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# In[23]:


df = pd.read_csv("../input/car-object-detection/data/train_solution_bounding_boxes (1).csv")
df.head()


# In[24]:


df['bbox_width'] = df['xmax']-df['xmin']
df['bbox_height'] = df['ymax']-df['ymin']

df['area'] = df['bbox_width']*df['bbox_height']


# In[25]:


# group by similar image names
df = df.groupby('image').agg(list)
df.reset_index(inplace=True)


# In[26]:


train_df, val_df = train_test_split(df, test_size=0.1, shuffle=False)
train_df.reset_index(inplace=True)
val_df.reset_index(inplace=True)


# In[27]:


img_root_path = Path("../input/car-object-detection/data/training_images")
                     
sample = train_df.iloc[215]
img_name = sample['image']
bboxes = sample[['xmin', 'ymin', 'xmax', 'ymax']].values


# In[28]:


img = cv2.imread(str(img_root_path/img_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0

bboxes = tuple(map(torch.tensor, zip(*bboxes)))
bboxes = torch.stack(bboxes, dim=0)


# In[29]:


labels = torch.ones(len(bboxes), dtype=torch.int64)


# In[30]:


transforms = A.Compose([
    A.Resize(256, 256, p=1.0),
    ToTensorV2(p=1.0),
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# In[31]:


augmented = transforms(image=img, bboxes=bboxes, labels=labels)


# In[32]:


torch.stack(tuple(map(torch.tensor, zip(*augmented['bboxes'])))).permute(1, 0).type(torch.float32)


# In[33]:


bboxes = map(torch.tensor, zip(*augmented['bboxes']))
bboxes = tuple(bboxes)
bboxes = torch.stack(bboxes, dim=0)


# In[34]:


img = augmented['image'].type(torch.float32)
bboxes = bboxes.permute(1, 0).type(torch.float32)


# In[35]:


area = sample['area']
iscrowd = torch.zeros(len(bboxes), dtype=torch.int)

target = {}
target['boxes'] = bboxes
target['labels'] = labels
target['area'] = torch.as_tensor(area, dtype=torch.float32)
target['iscrowd'] = iscrowd


# In[36]:


class LoadDataset(Dataset):
    def __init__(self, df, img_dir, transforms):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        # read & process the image
        filename = self.df.loc[idx, 'image']
        img = cv2.imread(str(self.img_dir/filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        
        # get the bboxes
        bboxes = self.df.loc[idx, ['xmin', 'ymin', 'xmax', 'ymax']].values
        bboxes = tuple(map(torch.tensor, zip(*bboxes)))
        bboxes = torch.stack(bboxes, dim=0)
        
        # create labels
        labels = torch.ones(len(bboxes), dtype=torch.int64)
        # apply augmentations
        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        
        # convert bbox list to tensors again
        bboxes = map(torch.tensor, zip(*augmented['bboxes']))
        bboxes = tuple(bboxes)
        bboxes = torch.stack(bboxes, dim=0)
        
        img = augmented['image'].type(torch.float32)
        bboxes = bboxes.permute(1, 0).type(torch.float32)
        iscrowd = torch.zeros(len(bboxes), dtype=torch.int)
        
        # bbox area
        area = self.df.loc[idx, 'area']
        torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        return img, target


# In[37]:


train_ds = LoadDataset(train_df, img_root_path, transforms)
val_ds = LoadDataset(val_df, img_root_path, transforms)


# In[38]:


model = fasterrcnn_resnet50_fpn(pretrained=True)

# get input features of classification layer
in_features = model.roi_heads.box_predictor.cls_score.in_features

# modify the classifier layer with number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)


# In[39]:


class ObjectDetector(LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.batch_size = 16
        self.model = self.create_model()
        
    def create_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        return model

    def forward(self, x):
        return self.model(x)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        return DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss_dict = self.model(inputs, targets)
        complete_loss = sum(loss for loss in loss_dict.values())
        
        self.log("train_loss", complete_loss, prog_bar=True)
        return {'loss': complete_loss}

    def validation_step(self, batch, batch_idx):
            inputs, targets = batch
            outputs = self.model(inputs)
            # calculate IOU and return the mean IOU
            iou = torch.stack(
                [box_iou(target['boxes'], output['boxes']).diag().mean() for target, output in zip(targets, outputs)]
            ).mean()
            
            return {"val_iou": iou}

    def validation_epoch_end(self, val_out):
        # calculate overall IOU across batch
        val_iou = torch.stack([o['val_iou'] for o in val_out]).mean()
        self.log("val_iou", val_iou, prog_bar=True)
        return val_iou


# In[53]:


detector_model = ObjectDetector()
trainer = Trainer(
    accelerator='auto',
    devices=1,
    auto_lr_find=True,
    max_epochs=5,
)

trainer.tune(detector_model)
trainer.fit(detector_model)


# In[68]:


import matplotlib.pyplot as plt

sample = val_ds[14]
img = sample[0]

detector_model.eval()
with torch.no_grad():
    out = detector_model([img])
    
# convert to numpy for opencv to draw bboxes
img = img.permute(1, 2, 0).numpy()


# In[69]:


# predicted bounding boxes    
pred_bbox = out[0]['boxes'].numpy().astype(int)
pred_label = out[0]['scores']

# draw bounding boxes on the image
for bbox, label in zip(pred_bbox, pred_label):
    # check if the predicted label is for car
    if label>=0.5:
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (255, 0, 0), thickness=2,
        )
    
plt.figure(figsize=(16, 6))
plt.imshow(img)


# In[ ]:





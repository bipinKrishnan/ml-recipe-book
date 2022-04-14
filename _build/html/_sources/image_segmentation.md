# Image segmentation

In the last chapter, we trained a model to classify animal images using pytorch lightning as our framework. In this chapter, instead of classifying the images into a category, we will segment roads present in the image as shown below:

```{image} ./assets/img_seg_pic.png
:alt: image_segmentation
:class: bg-primary mb-1
:align: center
```

As you can see the the segmented part of road is shown in green color. 

If we are segmenting roads in an image, what the models is trying to do is categorize whether each pixel in the input image belongs to a road or not:

```{image} ./assets/img_seg_model.png
:alt: image_segmentation
:class: bg-primary mb-1
:align: center
```

As you can see, during prediction, the pixels that belong to the road should be equal to 1 and all other pixels should be equal to 0.

### Dataset

The dataset we will be using for road segmentation can be found [here](https://www.kaggle.com/datasets/sakshaymahna/kittiroadsegmentation). This is how the directory structure of our dataset looks like:

```bash
kittiroadsegmentation
├── testing
└── training
    ├── calib
    ├── gt_image_2
    └── image_2
```

All the data that we will be using for this chapter is inside the 'training' folder. The training folder has two sub-folders(apart from ```calib``` folder), ```gt_image_2``` and ```image_2``` which contains our segmentation mask and its corresponding images respectively.

Here are some sample images and its segmentation masks from the above folders:

```{image} ./assets/seg_dataset.png
:alt: image_segmentation
:class: bg-primary mb-1
:align: center
```

#### Prepring the dataframe

Now let's write some code create a dataframe where each row contains the path to images and its corresponding segmentation masks.

```{note}
For a file inside ```image_2``` folder with the name 'um_000000.png', its corresponding image mask will have the name 'um_lane_000000.png'. We will expoit this pattern to extract all image paths and its masks.
```

First let's retrieve all the image file names from ```image_2``` folder
```python
from pathlib import Path

img_root_path = Path("../input/kittiroadsegmentation/training/image_2")
mask_root_path = Path("../input/kittiroadsegmentation/training/gt_image_2")

# all image file names are retrieved
img_files = img_root_path.glob('*')
```

Now let's loop through each image file name and check if it's segmentation mask file is also present:

```python
import os

def get_existing_imgs_and_masks(img_files, mask_root_path):
    existing_imgs, existing_masks = [], []
    
    for img_file in img_files:
        mask_file = f"{mask_root_path}/um_lane_{str(img_file).split('_')[-1]}"

        if os.path.exists(mask_file):
            existing_imgs.append(img_file)
            existing_masks.append(mask_file) 
         
    return existing_imgs, existing_masks
```

Call the function and see if it's working as intended:

```python
imgs, masks = get_existing_imgs_and_masks(img_files, mask_root_path)
```

It's time to put both ```imgs``` and ```masks``` into a dataframe:

```python
import pandas as pd

df = pd.DataFrame(columns=['imgs'])

df['imgs'] = imgs
df['masks'] = masks
```

Now split the dataframe into training and evaluation sets:

```python
from sklearn.model_selection import train_test_split

train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
```

#### Loading in images and masks

Finally, we are ready to write our dataset loading class using pytorch. These are the steps we will follow while loading our images and masks:

* Get the image path and its mask file path

```python
sample_img_path = str(imgs[0])
sample_mask_path = str(masks[0])
```

* Load the image using opencv, convert from BGR to RGB format and normalize the image by dividing it by 255.

```python
import cv2

sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)/255.0
```

* For the segmentation mask we don't need all the 3 channels, so we load in our mask using opencv, normalize it and take only the first channel:

```python
sample_mask = cv2.imread(sample_mask_path)/255.0
sample_mask = sample_mask[:, :, 0]
```

* Now we will make sure that the segmentation mask has only 1's and 0's(1 for pixels belonging to road and 0 for others).

```python
sample_mask = (sample_mask==1).astype(float)
```

* Now we will resize the image and mask and convert them to pytorch tensors. For this, we will use the '[albumentations](https://albumentations.ai/)' library which is commonly used by the machine learning community.

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# resize and convert to tensors
transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
augmented = transform(image=sample_img, mask=sample_mask)
```

* Right now, the size of our image and masks are (3, 256, 256) and (256, 256) respectively. We need to convert our mask to (1, 256, 256) and the data type of our image to ```FloatTensor```. Otherwise, we will get errors while training.

```python
import torch

augmented['mask'] = augmented['mask'].unsqueeze(0)
augmented['image'] = augmented['image'].type(torch.FloatTensor)
```

And that's it, we will wrap all of the above steps into our data loading class:

```python
from torch.utils.data import Dataset

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
```

Now let's load our training and evaluation datasets using the above class:

```python
train_ds = LoadDataset(train_df['imgs'].values, train_df['masks'].values)
eval_ds = LoadDataset(eval_df['imgs'].values, eval_df['masks'].values)

print(train_ds[0], eval_ds[0])
```

Output:

```python
{'image': tensor([[[0.0667, 0.1212, 0.2803,  ..., 0.0370, 0.0405, 0.0553],
        [0.1028, 0.1541, 0.3622,  ..., 0.0431, 0.0483, 0.0556],
        [0.1669, 0.1788, 0.3784,  ..., 0.0417, 0.0533, 0.0492],
        ...,
        [0.1184, 0.1194, 0.0632,  ..., 0.0609, 0.1167, 0.0814],
        ...
        ...
'mask': tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]], dtype=torch.float64)}
```

Wohoo! it's working without any errors.


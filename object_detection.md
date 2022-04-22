# Object detection

### Introduction

In this chapter we will talk about another task involving images as inputs, we will train a model to detect all objects in a given image, as shown in the below image:

```{image} ./assets/object_detect_img.png
:alt: object_detection
:class: bg-primary mb-1
:align: center
```

For this chapter we will train a model to detect cars in a given image. As shown above, the model should correctly detect all the cars in a given image by predicting the coordinates of the bounding box for the car.

```{image} ./assets/object_detect_model.png
:alt: object_detection
:class: bg-primary mb-1
:align: center
```

### Dataset

The dataset we will be using for training a car detecting model can be found [here](https://www.kaggle.com/datasets/sshikamaru/car-object-detection). All the images for our training set is under ```training_images``` folder and its bounding box coordinates can be found in ```train_solution_bounding_boxes (1).csv```. This is how the csv containing bounding box coordinates look like:

```{image} ./assets/obj_detect_ds.png
:alt: object_detection
:class: bg-primary mb-1
:align: center
```

The first column represents the name of the image and the remaining 4 columns represent the coordinates of the bounding box for that image. The columns ```xmin```, ```ymin```, ```xmax```, ```ymax``` represents ```x1```, ```y1```, ```x2```, ```y2``` coordinates resepctively. 

#### Preparing the dataframe

First let's read the csv file:

```python
import pandas as pd

df = pd.read_csv("../input/car-object-detection/data/train_solution_bounding_boxes (1).csv")
```

In the above dataframe, there are multiple rows with the same image name as shown below:

```{image} ./assets/obj_detect_multi_row.png
:alt: object_detection
:class: bg-primary mb-1
:align: center
```

This means that there are certain images that has multiple bounding boxes(for multuple cars), here is an image which has multiple rows in the dataframe:

```{image} ./assets/obj_detect_multi_img.png
:alt: object_detection
:class: bg-primary mb-1
:align: center
```

Let's create a column that contains the area of the bounding boxes, for that we need the height and width of the bounding boxes:

```python
df['bbox_width'] = df['xmax']-df['xmin']
df['bbox_height'] = df['ymax']-df['ymin']
```

Now let's calculate the area and store it in a new column:

```python
df['area'] = df['bbox_width']*df['bbox_height']
```

We will group all rows with the same image name together, thus, we will get all the bounding boxes for an image from a single row:

```python
# group by similar image names
df = df.groupby('image').agg(list)
df.reset_index(inplace=True)
```

Since the dataframe is ready, let's split it into training and validation sets:

```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.1, shuffle=False)
train_df.reset_index(inplace=True)
val_df.reset_index(inplace=True)
```

#### Loading the images and bounding boxes

Now we will write a data set loading class with pytorch. We will pass the dataframe(either training or validation), root path where the images are present and some augmentations/transforms to apply to the image.

Before writing the whole dataset loading class, let's take a sample image and it's bounding boxes from the dataframe to see what is really happening after each step in the data loading class.

* First let's take a sample image containing multiple bounding boxes:

```python
sample = train_df.iloc[215]
img_name = sample['image']
bboxes = sample[['xmin', 'ymin', 'xmax', 'ymax']].values
```

* We will read the image using opencv, convert it to RGB format and normalize it:

```python
import cv2
from pathlib import Path

# root path where images are present
img_root_path = Path("../input/car-object-detection/data/training_images")

img = cv2.imread(str(img_root_path/img_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
```

* Since our bounding box coordinates are in a list format, let's convert all of them into pytorch tensor and store it in tuple format:

```python
import torch

bboxes = tuple(map(torch.tensor, zip(*bboxes)))
print(bboxes)
```
Output:
```python
(tensor([  0.0000, 194.8600,  42.5557, 232.5177]),
 tensor([ 34.7294, 191.4366,  84.6223, 226.6490]),
 tensor([179.5166, 202.6850, 252.8886, 230.5614]),
 tensor([197.1259, 192.9038, 319.9016, 233.9849]),
 tensor([364.4139, 181.6554, 454.4168, 219.8021]),
 tensor([473.4935, 185.5679, 573.2793, 227.1380]),
 tensor([563.4964, 177.2539, 652.5210, 213.9334]))
```

Each tensor represents the xmin, ymin, xmax and ymax of a bounding box. So here we have 7 bounding boxes in total.

* Now we will stack all the above tensors to form a single tensor of shape (7, 4):

```python
bboxes = torch.stack(bboxes, dim=0)
print(bboxes)
```
Output:
```python
tensor([[  0.0000, 194.8600,  42.5557, 232.5177],
        [ 34.7294, 191.4366,  84.6223, 226.6490],
        [179.5166, 202.6850, 252.8886, 230.5614],
        [197.1259, 192.9038, 319.9016, 233.9849],
        [364.4139, 181.6554, 454.4168, 219.8021],
        [473.4935, 185.5679, 573.2793, 227.1380],
        [563.4964, 177.2539, 652.5210, 213.9334]])
```

* Similar to classification models, the object detection model we are going to use(faster rcnn with resnet50 backbone) requires the class/label to which each bounding box belongs to. So corresponding to each bounding box in our image, we should provide a label. In our case we only have bounding boxes for car, so we only have one label. The label 0 is reserved for background, so we will use 1 as the label for car. 

Let's create a tensor containing 1s(car label) that has the same length as the number of bounding boxes in the image:

```python
labels = torch.ones(len(bboxes), dtype=torch.int64)
```

* So, let's define the transforms/augmentations that we wish to apply to the image. We will use the 'albumentations' library for this. We will resize the image and convert it to tensors. 

Since this is an object detection task, we cannot simply resize the image without resizing the bounding boxes. But albumentations will automatically take care of this if we pass the ```bbox``` parameter. We should pass the bounding box format we are using as well as the key to store the labels.

The bounding format we are using is ```(xmin, ymin, xmax, ymax)``` which is the pascal voc format. 

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

transforms = A.Compose([
    A.Resize(256, 256, p=1.0),  # resize the image
    ToTensorV2(p=1.0),
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})  # bounding box parameters
```

Let's apply the transforms to our data:

```python
augmented = transforms(image=img, bboxes=bboxes, labels=labels)
```

The output will be a dictionary containing labels, augmented image and bounding boxes. 

* After augmentation, the bounding boxes are stores in a list as tuples. But our model expects them as stacked tensors as we did earlier(before augmentation). So, we have to again make the bounding boxes in that format.

For that we need to convert each bounding box coordinate list to tensors, store them in a tuple abd then stack them to a single tensor:

```python
# convert to tensor
bboxes = map(torch.tensor, zip(*augmented['bboxes']))
# store as tuples
bboxes = tuple(bboxes)
# stack into a single tensor
bboxes = torch.stack(bboxes, dim=0)
```

* Now let's convert the data types to the format as required by the model(otherwise we will get errors while training):

```python
img = augmented['image'].type(torch.float32)
bboxes = bboxes.permute(1, 0).type(torch.float32)
```

* Our image and the bounding boxes are ready. Our model expects some more elements apart from the bounding boxes as the targets, which includes, 'labels', 'area' and 'iscrowd'.

We are familiar with 'labels' and 'area'(area of the bounding box). But what is this 'is crowd' element?

Similar to 'labels', we should have a value for 'iscrowd' corresponding to each bounding box. If 'iscrowd' is 1, that bounding box is not considered by the model. But we want the model to consider all bounding boxes. So we will put the value 0 corresponding to each bounding box:

```python
iscrowd = torch.zeros(len(bboxes), dtype=torch.int)
```

* Since everything is ready lets create a dictionary and store all the target data there:

```python
area = sample['area']

target = {}
target['boxes'] = bboxes
target['labels'] = labels
target['area'] = torch.as_tensor(area, dtype=torch.float32)
target['iscrowd'] = iscrowd
```

```{note}
Make sure that the keys of the target dictionary has the same name as above, because the model expects it that way.
```

Now let's wrap everything into our data set loading class:

```python
from torch.utils.data import Dataset

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
        area = sample['area']
        torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        return img, target
```

Finally, let's load the training and validation dataset:

```python
train_ds = LoadDataset(train_df, img_root_path, transforms)
val_ds = LoadDataset(val_df, img_root_path, transforms)
```

### Training the model



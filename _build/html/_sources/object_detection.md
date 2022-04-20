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

#### Loading the images and bounding boxes
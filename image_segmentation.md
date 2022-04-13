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


## Annotations, Labels, and Patches

1.  Annotations are the 'marked up' images with pixels of each object assigned a unique pixel index.  

2.  Labels are the annotations that we want to use for training.  These can be indicated by drawing a bounding box around them.  Note:  The reason to have both 'annotations' and 'labels' is that the annotations may not be complete.  This image set is a great one to illustrate this.  We have 140 images, and 100+ annotated, but some of the annotations do not have all the protrusions labeled.  So for this problem we can use bounding boxes to mark the images (or areas of images) for which annotations are mature. 

3.  Patches - Patches are an artifact created by cropping and augmenting the labels.  From one label we can generate hundred of patches (cropped, stretched, color augmented and intensity augmented variations of the label).  Note that in my workflows I like to save the augmentations for troubleshooting.  In many DL training toolkits augmentations is done as part of the training cycle (for example there is already augmentation build into Cellpose).

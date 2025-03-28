## Detecting vessels in 3D light sheet image using slice by slice 2D semantic segmentation UNET.

This examples show how to train a 2D semantic segmentation model on a few ROIs on a few slices from a very large 3D light sheet image.  We then apply the model on every slice of the image and visualize in Napari. 

Data is from this [question](https://forum.image.sc/t/vessel-segmentation-in-3d-or-2d-with-ml-or-dl/110263/6)

Note:  A 3D segmentation model may work better (though will take more memory).  

This example shows

1.  How to perform sparse labelling. 
2.  How to use the napari-easy-augment-batch-dl plugin to label a few ROIs in a set of 2D images, and how to supplement a small number of labels with augmentation. 
3.  How to code a semantic segmentation unet using pytorch.  I provide the code but also show how to train the unet in the Napari GUI. 
4.  How to apply a 2D model to every slice of a large 3D image using monai and sliding_window_inference.
5.  How to use Napari to visualize the final labels and troubleshoot.  
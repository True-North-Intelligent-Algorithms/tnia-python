##  SEM grain detection with Zeiss image set

This sequence of notebooks is an attempt to replicate figure 8 in this [article](https://academic.oup.com/mt/article/32/6/13/7922218?login=false).  Also see [this Question](https://forum.image.sc/t/free-article-from-machine-learning-to-deep-learning-revolutionizing-microscopy-image-analysis/106340).
 

The images in question are SEM images of polycrystalline materials.  To count the grains there are a few approaches. 

1.  Classical image processing (thresholding, opening, closing, watershed, etc.)
2.  Machine learning pixel segmentation
3.  Deep learning semantic (pixel) segmentation
4.  Deep learning instance segmentation

Notes for cellpose instance segmentation. 

1.  Carefully relabel an ROI of small and large grains.  
2.  Turn rescale 'off' during training. 
3.  During evaluation change ```niter``` to 2000
4.  Change ```bsize``` to a larger value (512?) for training and evaluation. 
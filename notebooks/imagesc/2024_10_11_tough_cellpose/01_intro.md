## Challenging Segmentation with Cellpose

This sequence of notebooks is an attempt to improve segmentation for [this Question](https://forum.image.sc/t/challenging-segmentation-with-cellpose-need-help/103618).
 
[Arkajyoti Sarkar)[https://www.linkedin.com/in/arkaj/?originalSubdomain=in] and his group have kindly shared many images with the community which you can find a copy of [here](https://www.dropbox.com/scl/fo/y1a9y80mi843xslhsqn73/AIiUn9oGqVjnDzew2shu66U?rlkey=j64363njxo6vvo2azvkl32dau&st=yvp1wp64&dl=0).  

The images in question are large Cells with long thin protrusions.  Cellpose in default training mode resizes images, so it is possible that the downsizing may distort thin features.  However there are confounding complications.  For example the long thin protrusions are difficult to see and trace sometimes, so some of the labels may be missing protrusions.   I attempted to training a Cellpose model for these images, and had some success detecting more protrusions. 

**FIRST OFF CRITICAL NOTE**  My goal was to determine what Cellpose settings are needed to capture the long thin features.  I did not rigorously test my models to see if they would generalize.  

Here are the various strategies I found helpful for this image set. 

1.  Carefully relabel more of the long protrusions in the label images. 

In my test the above was most critical.  Even with default Cellpose settings I obtained better protrusion detection after some relabeling. 

2.  Turn rescale 'off' during training. 

This did help detect some of the thinner protrusions. 

3.  During evaluation change ```niter``` to 2000

Sometimes obvious thicker protrusions were missing.  There is a parameter called ```niter``` which sets the number of iterations used when simulating dynamics and determining pixels that converge to the same position.  To detect longer protrusions I found a higher ```niter``` was needed. 

4.  Change ```bsize``` to 512 for training and evaluation

This seems to have helped a bit, but may not have been critical.  
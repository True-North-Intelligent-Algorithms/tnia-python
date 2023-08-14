# Installing OpenCV for c++ and java development

There are Python wrappers for opencv that can easilly be installed into a python environment using pip or conda.  However for c++ and/or java development opencv needs to be installed independantly.   The below instructions are some hints and reminders (mostly for myself) for installing opencv. 

## Windows

Pre-built OpenCV releases can be found here https://opencv.org/releases/.  The installer doesn't seem to install the components to a pre-determined location (like 'c:\Program Files') but just extracts them from the location it is run, so be careful to copy the OpenCV components somewhere, where they can be found later.

I have been copying them to 'c:\Program Files\' 

## Linux

Seems there is not an opencv release for linux so need to build it.  See below

https://www.samontab.com/web/2020/11/installing-opencv-4-5-0-in-ubuntu-20-04-lts/

Use ccmake to turn some options off.  

It may help to turn opencv-dnn option off if not needed, some systems have trouble compiling this

Need to make sure Ant is installed for Java wrappers

May also help to turn 'WITH_OPEN_JPEG' off. 

When deploying not all linux systems have FFMPEG and other dependencies in stalled, so also may help to turn off

WITH_1394
WITH_ADE
WITH_EIGEN
WITH_FFMPEG

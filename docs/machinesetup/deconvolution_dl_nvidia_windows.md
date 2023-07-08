---
layout: basic
---

## Setup a NVIDIA deconvolution/deep learning machine for Windows

These instructions are for setting up an NVIDIA deconvolution/deep learning machine on Windows appropriate for running and modifying the tnia-python deconvolution/convolution and deep learning examples and utilities.   Some of the instructions are specific to AWS and can be ignored if running on physical hardware.

Note:  Napari may not be supported using Remote Desktop Protocol (RDP) thus you may not be able to start Napari when using an AWS EC2 VM over RDP. 

### (If running on AWS Windows Servers) Disable IE enhanced security configuration and (optionally) install Chrome.

If starting a new AWS Windows Servers machine the default web browser will be Microsoft Explorer with security levels set so high it is unusable.  Open the Server Manager (Start > Server Manager). In the Local server section, disable the  "Internet Explorer Enhanced Security Configuration".  Instructions may be slightly different for different version of Windows Server.  Google "disable internet explorer enhanced security" for more info. 

After disabling IE enhanced security optionally install Chrome. 

### Install Nvidia drivers

Install the newest nvidia drivers from [here](https://www.nvidia.com/download/index.aspx).   You will be asked for product type and series.  

For AWS ec2 machines the AWS G5 series or the P3 series are often used for high performance GPU processing and machine learning.   Double check what type of GPU your instance is using.  G5 series uses Tesla A10G.  

You will have to choose Cuda Tooklit.   Not sure if it matters what version...  

### Install Mini-Conda

See [here](https://docs.conda.io/en/latest/miniconda.html) 

### Install Git for Windows

https://gitforwindows.org/

## Install Visual Studio Code 

Install Visual Studio Code with the following extensions

Python
Jupyter

Optional (but helpful)
Vim
Github Copilot


## Install Mamba and devbio-napari

Run 'anaconda powershell' from Windows start menu.  

conda install mamba -c conda-forge
mamba create --name decon_dl python=3.9 devbio-napari -c conda-forge -c pytorch
mamba activate decon_dl


### Install Tensorflow for Windows Native


Run the following commands in the Anaconda powershell, the last command should show a list of GPUs.  

It is important you specify python version 3.9, cudatoolkit 11.2, cudnn 8.1.0 and tensorflow<2.11.  Newer version of tensorflow do not support GPU processing on 'Native' Windows (they only support GPU via (WSL (Windows Subsystem for Linux).)


```
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

### Install CUPY, stardist, psfmodels and clij2-fft

It is important to specify cuda11x to be compatible with cudatoolkit 11.2  

pip install cupy-cuda11x  
pip install stardist  
pip install psfmodels  
pip install --index-url https://test.pypi.org/simple/ --no-deps clij2-fft  



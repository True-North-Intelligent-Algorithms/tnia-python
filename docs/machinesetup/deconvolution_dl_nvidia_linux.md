## Setup a deconvolution/deep learning machine for Linux (Ubuntu)

### Install NVIDIA drivers

Use nvidia-smi to check if nvidia driver is installed, if not found you need to install drivers.

```
nvidia-smi  
```

Use ubuntu-driver to install, first llist to see which drivers are available  

```
sudo ubuntu-drivers list  
```

Then install  

```
sudo ubuntu-drivers install nvidia:525  
```

Full instruction [here](https://help.ubuntu.com/community/NvidiaDriversInstallation)

test363

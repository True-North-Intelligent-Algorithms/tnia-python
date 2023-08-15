---
layout: basic
---

## Notes for installing Cuda on Ubuntu

There are different ways to install.  Cuda can be installed by Debian package manager (apt), by downloading Cuda from NVIDIA's website.   

## Nvidia drivers

When installing cuda you also need to install the right nvidia drivers.  There are different ways to install Nvidia drivers, including package manager, downloading from NVIDIA's website, and using the ```ubuntu-drivers``` tool. 

Use 'nvidia-smi' command to check if driver is installed.  If installed a print-out of device and driver information will be shown. 

If Nvidia drivers are not installed by the NVIDIA cuda installer they can be installed with the 'ubuntu-drivers' tool.  Documentation can be found [here](https://help.ubuntu.com/community/NvidiaDriversInstallation)

Note:  When installing Cuda from the NVIDIA tool (.run file) it will suggest to uninstall current driver and re-install the driver through the NVIDIA tool.  I've found that it is OK to use previous driver (ie installed by apt or ubuntu-drivers) as long as this driver is compatible with the Cuda version.  And in fact you will want to do this if instalnstaling multiple versions of cuda. 

## Uninstalling NVIDIA driver and/or previous version of Cuda

Note:  Multiple versions of cuda are OK. 

When installing the Cuda tookkit from the .run file it will suggest uninstalling the previous driver and or previous cuda (if a previous driver is found)

You have to remember how you installed the driver and/or Cuda if uninstalling

1.  Note if uninstalling driver installation from apt
```
sudo apt-get remove --purge nvidia\*
sudo apt-get autoremove
```
  
2.  If uninstalling .run file installation from the Nvidia web site, run cuda-uninstaller in /usr/local/cuda-11.7/bin (or replace cuda-11.7 with appropriate version). 

See https://forums.developer.nvidia.com/t/how-to-uninstall-cuda-toolkit-and-cudnn-under-linux/47923/13

## Installing cuda from NVIDIA .run file

NVIDIA provides .run files for each version of Cuda.  I have been installing via the run file, and found it a good way to get multiple versions of cuda installed. 

For example to install cuda 11.8 

```
sudo ./cuda_11.8.0_520.61.05_linux.run
```

This will try to install driver 520.61 as well, however (if you have equivalent or newer driver) uncheck the driver installation and just install cuda.  

Then change your path to include this version of cuda by editing .bashrc

```
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$L  D_LIBRARY_PATH"  
```

Multipe version of cuda can exist under /usr/local/ and you can change LD_LIBRARY_PATH as to choose which version is used.

(Note: may have to be careful that a simlink ```/usr/local/cuda``` does not link to another version)

## Notes for getting Cuda to work with CMake on Ubuntu

1.  It seems like you need to be careful that all paths are set correctly.  This includes the following.  Be careful that CUDACXX points to the executable of the compiler, not just the path.  

 export PATH="/usr/local/cuda-11.7/bin:$PATH"  
 export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$L  D_LIBRARY_PATH"  
 export CUDA_PATH='/usr/local/cuda-11.7/'  
 export CUDACXX='/usr/local/cuda-11.7/bin/nvcc'  
 

---
layout: basic
---

## Notes for installing Cuda on Ubuntu

There are different ways to install.  Cuda can be installed by package manager (apt) or by downloading Cuda from NVIDIA's website.  

## Nvidia drivers

Use 'nvidia-smi' command to check if driver is installed.  If installed a print-out of device and driver information will be shown. 

If Nvidia drivers are not installed by the installer they can be installed with the 'ubuntu-drivers' tool.  Documentation can be found [here](https://help.ubuntu.com/community/NvidiaDriversInstallation)

## Uninstalling

To install the Cuda tookkit from the .run file it will suggest uninstalling the previous driver and or previous cuda (if a previous driver is found)

You have to remember how you installed the driver and/or Cuda if uninstalling

1.  Note if uninstalling driver installation from apt
```
sudo apt-get remove --purge nvidia\*
sudo apt-get autoremove
```
  
2.  If uninstalling .run file installation from the Nvidia web site, you need to extract and run the 'uninstaller'.

** Run the .run file with the --extract=ANY_ABSOLUTE_PATH to extract the uninstaller
** Then you can find the cuda-uninstaller in the above given path inside the ‘bin’ dir, run it and select which cuda versions you’d like to uninstall

See https://forums.developer.nvidia.com/t/how-to-uninstall-cuda-toolkit-and-cudnn-under-linux/47923/13

## Notes for getting Cuda to work with CMake on Ubuntu

1.  It seems like you need to be careful that all paths are set correctly.  This includes the following.  Be careful that CUDACXX points to the executable of the compiler, not just the path.  

 export PATH="/usr/local/cuda-11.7/bin:$PATH"  
 export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$L  D_LIBRARY_PATH"  
 export CUDA_PATH='/usr/local/cuda-11.7/'  
 export CUDACXX='/usr/local/cuda-11.7/bin/nvcc'  
 

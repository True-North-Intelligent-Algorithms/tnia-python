---
layout: basic
---

## Notes for installing Cuda on Ubuntu

1.  Note that there are different ways to install, and you have to remember how you installed Cuda if uninstalling.  Cuda can be installed by package manager (apt) or by downloading Cuda from NVIDIA's website.  
2.  If downloading from the web site, you need to extract and run the 'uninstaller'.

** Run the .run file with the --extract=ANY_ABSOLUTE_PATH to extract the uninstaller
** Then you can find the cuda-uninstaller in the above given path inside the ‘bin’ dir, run it and select which cuda versions you’d like to uninstall

See https://forums.developer.nvidia.com/t/how-to-uninstall-cuda-toolkit-and-cudnn-under-linux/47923/13

## Notes for getting Cuda to work with CMake on Ubuntu

1.  It seems like you need to be careful that all paths are set correctly.  This includes the following.  Be careful that CUDACXX points to the executable of the compiler, not just the path.  

 export PATH="/usr/local/cuda-11.7/bin:$PATH"  
 export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$L  D_LIBRARY_PATH"  
 export CUDA_PATH='/usr/local/cuda-11.7/'  
 export CUDACXX='/usr/local/cuda-11.7/bin/nvcc'  
 
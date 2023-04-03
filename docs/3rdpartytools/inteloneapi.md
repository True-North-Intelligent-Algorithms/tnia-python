---
layout: basic
---

## Intel One API

IntelOneAPI base toolkit can be found [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).

IntelOneAPI has windows and linux installers.  It should be installed in the directory below

```
C:\Program Files (x86)\Intel\oneAPI
```

Check if this directory exists to see if IntelOneAPI is on the system.  

If using CMake we have to set parameters to tell CMake where to find IntelOneAPI components.  Below is an example of setting such parameters when calling CMake from the command line. 

```
-DIPP_LIBRARY_DIR="C:/Program Files (x86)/Intel/oneAPI/ipp/latest/lib/intel64" \
-DIPP_INCLUDE_DIR="C:/Program Files (x86)/Intel/oneAPI/ipp/latest/include" \
-DMKL_INCLUDE_DIR="C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include" \
-DMKL_LIBRARY_DIR="C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64" \
```

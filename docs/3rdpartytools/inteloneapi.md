---
layout: basic
---

## Intel oneAPI

IntelOneAPI base toolkit can be found [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).

IntelOneAPI has windows and linux installers.  It should end up being installed in the directory below

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

## Linking Intel oneAPI

## Intel IPP

For IPP see [these instructions](https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-oneapi/2022-2/linking-options.html)

Note:  There are both static and dynamic versions of the library.  There are also non-threaded and threaded versions.  You need to use the table to choose the linking model and the threading model then link using the appropriate directory and libary names. 

## Intel MKL

As with IPP a number of choices have to be made (dynamic vs static linking, operating system, threading model).  Intel provides a link advisor page with drop down menus to aid with finding the correct linking see [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.u014x8)

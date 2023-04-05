---
layout: basic
---

## Boost

Boost provides free peer-reviewed portable C++ source libraries..  Boost can be found [here](https://www.boost.org/)

Boost needs to be extracted to a location on your system, then tools like CMake need to be given the location of the boost library.  

Often times, when installing a project that uses boost on a new system it isn't apparent where CMake (or other build tool) is searching for boost. 

Search the code base for 'boost' and look for a CMake input parameter something like below to determine where you should extract boost. 

```
-DBOOST_INCLUDE_DIR="C:\ProgramData\boost_1_81_0" \
```




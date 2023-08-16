---
layout: basic
---

## Linux trouble shotting

### to look at the dependencies of a library

```
ldd library_name.so
```

### to look at embedded path which the library is searching

```
readelf -d library_name.so | grep -i path 
```
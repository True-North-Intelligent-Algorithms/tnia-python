---
layout: basic
---

## Helpful grep commands

### Search recursively through files but look in only one file type

For example this command searches all cpp filtes recursively, for "fft32f"

```
 grep -r --include=\*.cpp "fft32f" .
```

### Search recursively through file look in multiple file types

Can achieve this by using the --include option multiple times

```
 grep -r --include=\*.java --include=\*.cpp "conv3d_32f" .
```

---
layout: basic
---

## Remove directories of the same name recursively

This is useful when cleaning a build.  For example when building using the ```cppbuild.sh``` scripts build directories called ```cppbuild``` are created in each sub-directory.  To get rid of them all

```
find . -type d -name cppbuild -exec rm -rf {} \;
```

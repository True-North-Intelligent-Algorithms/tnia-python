---
layout: basic
---

# Packaging 

A good overview of packaging is [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

# Packaging .dlls and libraries

One approach is to pass the library installation directory and the local library file locations to setup(...) with the data_files parameter.

```
setup(...,
data_files=[('lib/',['lib/win64/clij2fft.dll','lib/win64/clFFT.dll', 'lib/linux64/libclij2fft.so', 'lib/linux64/libclFFT.so.2'])],
...)
```

Note:  Data_files may be depracated, thus more research is needed into this topic.

## Generating distribution archives

You may need to install build, if it's not there already

```
pip install build
```

Go to the directory with setup.py in it and run


```
python -m build
```

## uploading to test server

Test server and pip server require different accounts.
Test server is [here](https://test.pypi.org/manage/projects/)

register for an account

* Create, copy and save an API token

then to upload

```
python -m twine upload --repository testpypi --skip-existing dist/*
```

It will prompt you for username (which is __token__ for some reason) and the password is the above API token.

--skip-existing is needed when upgrading package.  Only upload after testing thoroughly, it seems you can't overwrite existing versions.

## Install from test server

 pip install --index-url https://test.pypi.org/simple/ --no-deps clij2-fft





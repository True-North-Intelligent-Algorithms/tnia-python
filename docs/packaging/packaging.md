---
layout: basic
---

# Packaging 

A good overview of packaging is [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
(may be out of date)

# Versioning

Can be a bit confusing in newer projects with ```pyproject.toml``` files.

For example if you see 

```
dynamic = ["version"]
```

In a TOML file it means version is dynamically determined by another method.  Need to search for more ```version``` info. 

For exmaple

```
[tool.setuptools.dynamic]
version = {attr = "napari_easy_augment_batch_dl.__init__.__version__"}
```

Indicates the version is determined with an attribute in the ```napari_easy_augment_batch_dl```, ```__init__``` file.   And if we open that file we should see

```
__version__ = "0.0.4"
```

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
## uploading to real pip

```
python -m twine upload --skip-existing dist/*
```

It will prompt you for username (which is __token__ for some reason) and the password is the API token.

API tokens are something like ```pypi-alksdfjlkasdfjklajsdlkfjklasjfdlkas```, they begin with pypi so when you copy the token from your secret secure file make sure you copy the pypi part. 

--skip-existing is needed when upgrading package.  Only upload after testing thoroughly, it seems you can't overwrite existing versions.

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





# Packaging 

A good overview of packaging is [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## Generating distribution archives

Go to the directory with setup.py in it and run

```
python -m build
```

## uploading to test server

Test server and pip server require different accounts.
Test server is [here](https://test.pypi.org/manage/projects/)

register for an accout

* Create, copy and save an API token

then to upload

```
python -m twine upload --repository testpypi --skip-existing dist/*
```

It will prompt you for username (which is __token__ for some reason) and the above API token (password)

--skip-existing is needed when upgrading package.  Only upload after testing thoroughly, it seems you can't overwrite existing versions.

## Install from test server

 pip install --index-url https://test.pypi.org/simple/ --no-deps clij2-fft





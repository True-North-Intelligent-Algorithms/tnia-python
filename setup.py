from setuptools import setup, find_packages

setup(name='tnia-python',
      version='0.1',
      description='A collection of utilities',
      url='http://github.com/tnia/tnia-python',
      author='Brian Northan',
      author_email='bnorthan@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['SimpleITK-SimpleElastix', 'pyopencl', 'pyclesperanto-prototype', 'napari[all]'],
      zip_safe=False)

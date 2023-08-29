from setuptools import setup, find_packages

setup(name='tnia-python',
      version='0.1.18',
      description='A collection of utilities',
      url='http://github.com/tnia/tnia-python',
      author='Brian Northan',
      author_email='bnorthan@gmail.com',
      license='MIT',
      packages=find_packages(),
      #install_requires=['napari-plugin-engine','SimpleITK-SimpleElastix', 'pyclesperanto-prototype', 'napari[all]','rawpy','python-bioformats'],
      zip_safe=False,
      #package_data = {'tnia-python':['napari.yaml']},
      #entry_points={
      #  'napari.plugin': ['Napari Test = tnia.napari.napari_demo'],
      #  'napari.manifest': ['tnia-python = tnia.napari:napari.yaml']
      #},
)

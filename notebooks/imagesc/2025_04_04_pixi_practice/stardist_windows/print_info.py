import numpy
import tensorflow as tf 
import napari
import stardist
import sys

print('Python:', sys.version)
print('numpy is', numpy.__version__)
print('Tensorflow:', tf.__version__, '| GPU:', tf.test.is_gpu_available(), '| Name:', tf.test.gpu_device_name() if tf.test.is_gpu_available() else 'None')
print('napari:', napari.__version__)
print('stardist:', stardist.__version__)
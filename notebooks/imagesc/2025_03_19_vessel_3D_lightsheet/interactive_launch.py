# note have to import napari first and show the viewer
# otherwise on Linux I get an error if I show the viewer after
# importing Cellpose
import napari
viewer = napari.Viewer()

import os
from napari_easy_augment_batch_dl import easy_augment_batch_dl

# import the frameworks you want to use.  As part of the import the framework wil be registered
# if it calls the BaseFramework.register_framework() method
# right now most commented out except for vessels_semantic_framework
import vessels_semantic_framework

'''
try:
    from napari_easy_augment_batch_dl.frameworks.pytorch_semantic_framework import PytorchSemanticFramework
except:
    print('PytorchSemanticFramework not loaded')
'''

# create the napari-easy-augment-batch-dl widget, we pass import_all_frameworks=False because
# we already imported the frameworks we want to use
batch_dl = easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer, import_all_frameworks=False)

viewer.window.add_dock_widget(
    batch_dl
)

# here we list directories for image sets we have been experimenting with
# need to point the parent_path to a directory with the 2D image set we want to work with

test_dataset = "None"

#!!!!! Change to local directory 
parent_path = r'D:\images\tnia-python-images\imagesc\2025_03_19_vessel_3D_lightsheet'

model_path = os.path.join(parent_path, r'models')

batch_dl.load_image_directory(parent_path)
       
napari.run()
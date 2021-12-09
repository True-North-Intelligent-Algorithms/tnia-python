from napari_plugin_engine import napari_hook_implementation
from .simple import simple_plugin

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [simple_plugin]

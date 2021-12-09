# Napari Plugins

## add a dock widget plugin

[Overview for Plugin Developers](https://napari.org/plugins/stable/for_plugin_developers.html#plugins-for-plugin-developers)

Need to add a module that implements napari_hook_implementation napari_experimental_provide_dock_width as follows 

```
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [simple_plugin]
```

Then need to add the entry point to either setup.py or setup.cfg.  The entry point is the module containing the hook

'''
entry_points={
        'napari.plugin': ['Napari Test = {packagename}.{modulename}'],
      }
'''

(Note in some examples only the packagename is used, and the hook function is imported in the __init__.py file of the package.)  
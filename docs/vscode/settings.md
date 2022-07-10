---
layout: basic
---

## Add vscode settings file

In tnia coded we've been using the settings file to pass cmake library and include paths.  The settings file is only generated when a setting is changed.  Thus to create the file you need to change a setting from the GUI.  At that point the file will be generated and additional settings can be added manually. 

1.  In VS Code go to 'file->preferences->settings'
2.  Go to workspace setting.
3.  In the settings dialog choose the 'Workspace' tab (otherwise settings will be set for the User)
4.  Go to the Extensions category on the left pane, then highlight CMakeTools. 
5.  Scroll down the cmake options until you find ```Cmake: Configure Settings```.
6.  Click on ```Edit in settings.json```
7.  The settings.json chould be generated in a ```.vscode``` folder at the top level of the workspace. 

From there we can manually edit the settings file.  The file is for all settings not just cmake settins.  

Below is an example file that shows the basic format, including the ```cmake.configureSettings``` section, which is very useful for setting up cmake paths on a local machine. 

```
{
    "cmake.configureSettings": {
        "OPENCL_INCLUDE_DIR":"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include",
        "CLIC_INCLUDE_DIR":"C:/Users/bnort/work/ImageJ2022/clij/CLIc_prototype/clic/include/core",
        "OCLCL_INCLUDE_DIR":"C:/Users/bnort/work/ImageJ2022/clij/CLIc_prototype/thirdparty/opencl/ocl-clhpp/include",
        "CLIC_LIBRARY_DIR":"C:/Users/bnort/work/ImageJ2022/clij/CLIc_prototype/bin/lib",
        "OPENCV_INSTALL_DIR":"C:/Program Files/opencv/build",
    },
    "files.associations": {
        "vector": "cpp",
        "__hash_table": "cpp"
    }
}
```
import napari

viewer = napari.Viewer()

from tnia.napari.deconvolution import restoration_deconvolution
viewer.window.add_dock_widget(restoration_deconvolution.RestorationDeconvolutionPlugin(viewer))

k=input("press close to exit") 
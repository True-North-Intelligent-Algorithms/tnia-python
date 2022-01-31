from napari_plugin_engine import napari_hook_implementation
from skimage.filters.thresholding import threshold_local
from tnia.napari.qtdesigner_demo.simple_threshold import Ui_Form 
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from skimage.filters import threshold_otsu

# derive from UI_Form, a form designed with qtdesigner
class threshold_widget(QWidget, Ui_Form):
    def __init__(self, napari_viewer, parent=None):
        QWidget.__init__(self, parent)
        self.setupUi(self)
        self.viewer=napari_viewer
          
        self.pushButtonThreshold.clicked.connect(self.threshold_pushed)
        self.viewer.bind_key('t', self.threshold_pushed_from_key)

    def threshold_pushed_from_key(self, viewer):
        self.threshold_pushed()

    def threshold_pushed(self):
        print('NUM LAYER ARE',len(self.viewer.layers))

        # layer to process (could use active, but that will return nothing if 
        # there are multiple layers selected, this deselects and returns the first selected) 
        layer= self.viewer.layers.selection.pop()
    
        if (self.radioButtonGlobal.isChecked()==True):
            print('GLOBAL')
            global_thresholded = layer.data>threshold_otsu(layer.data)
            temp=self.viewer.add_image(global_thresholded)
            self.viewer.layers.selection.remove(temp)
        if (self.radioButtonAdaptive.isChecked()==True):
            print('LOCAL')
            local_thresholded = layer.data>threshold_local(layer.data, 11)
            temp=self.viewer.add_image(local_thresholded)
            self.viewer.layers.selection.remove(temp)

        # reselect the layer    
        layer.selected=True


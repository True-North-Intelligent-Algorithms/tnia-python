from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

class simple_plugin(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.layout().addWidget(btn)
    
    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog

class NapariImageChooserWidget(QWidget):
    def __init__(self, viewer=None, parent=None, button_text="Open File", show_button=False):
        QWidget.__init__(self, parent=parent)
        self.viewer = viewer

        self.layout = QHBoxLayout(self)

        if show_button is True:        
            # button to open file dialog
            self.left_widget = QPushButton(button_text, self)
            self.layout.addWidget(self.left_widget)
            self.left_widget.clicked.connect(self.open_image)
        else:
            self.left_widget = QLabel(button_text, self)
            self.layout.addWidget(self.left_widget)
        # combo box to choose napari image
        self.napari_image_combo_box = QComboBox(self)
        self.layout.addWidget(self.napari_image_combo_box)

        self.setLayout(self.layout)

        self.viewer.layers.events.changed.connect(self.update_combo_box)
        self.viewer.layers.events.inserted.connect(self.update_combo_box)

    def open_image(self):
        options = QFileDialog.Options()

        # Open a file dialog to select a file
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', options=options)

        # If a file was selected, add it to the combo box
        if file_name:
            layers = self.viewer.open(file_name)
            self.napari_image_combo_box.addItem(layers[0].name)
            self.napari_image_combo_box.setCurrentIndex(self.napari_image_combo_box.count() - 1)

    def update_combo_box(self):
        self.napari_image_combo_box.clear()
        for layer in self.viewer.layers:
            self.napari_image_combo_box.addItem(layer.name)


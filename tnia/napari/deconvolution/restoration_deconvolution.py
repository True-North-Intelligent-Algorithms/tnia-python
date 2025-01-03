from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QFileDialog, QLabel, QProgressBar, QStyledItemDelegate, QCheckBox, QSizePolicy, QPushButton
from qtpy.QtWidgets import QMessageBox, QComboBox, QAbstractItemView
from qtpy.QtGui import QPainter
from qtpy.QtCore import Qt
from tnia.gui.deconvolution.instrument_ui import Ui_Form as instrument_UI_Form
from tnia.gui.deconvolution.richardson_lucy_ui import Ui_Form as richardson_lucy_UI_Form
# import QThread to pause the program
from qtpy.QtCore import QThread
import json
from tnia.deconvolution.psfs import gibson_lanni_3D, psf_from_beads
from tnia.models.instrument_models import InstrumentModelKeys, write_psf
from tnia.gui.widgets.napari_image_chooser_widget import NapariImageChooserWidget
from tnia.gui.widgets.image_batch_widget import ImageBatch

class InstrumentWidget(QWidget, instrument_UI_Form):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)  # Adjust the minimum size as needed
        self.comboBox_modality.addItems(["Widefield", "Confocal", "2P", "STED", "Lattice Light Sheet"])

class RichardsonLucyWidget(QWidget, richardson_lucy_UI_Form):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 100)  # Adjust the minimum size as needed
        self.setMaximumSize(500, 110)  # Adjust the minimum size as needed

class RestorationDeconvolutionPlugin(QWidget):
    def __init__(self, napari_viewer, parent=None):
        QWidget.__init__(self, parent=parent)
        self.viewer = napari_viewer
        self.instrument_models_path = ""
        #self.setupUi(self)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.setLayout(self.main_layout)

        self.open_instrument_models_folder = QPushButton('Open Models Folder', self)
        self.label_instrument_models_folder = QLabel("Models Folder")
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.open_instrument_models_folder) 
        h_layout.addWidget(self.label_instrument_models_folder)
        self.main_layout.addLayout(h_layout)

        self.label_select_psf = QLabel("Select Model")
        self.combo_box_select_psf = QComboBox(self)
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.label_select_psf)
        h_layout2.addWidget(self.combo_box_select_psf)
        self.main_layout.addLayout(h_layout2)

        self.open_instrument_models_folder.clicked.connect(self.open_instrument_models_folder_clicked)
        self.main_layout.addWidget(QLabel("Instrument Parameters"))
        
        self.instrument = InstrumentWidget(self)
        self.instrument.doubleSpinBox_xy_spacing.setValue(0.1)
        self.instrument.doubleSpinBox_z_spacing.setValue(0.2)
        self.instrument.spinBox_xy_size.setValue(128)
        self.instrument.spinBox_z_size.setValue(64)
        self.instrument.doubleSpinBox_NA.setValue(1.4)
        self.instrument.doubleSpinBox_wavelength.setValue(0.5)
        self.instrument.doubleSpinBox_imm_RI.setValue(1.5)
        self.instrument.doubleSpinBox_sample_RI.setValue(1.33)
        self.instrument.doubleSpinBox_depth.setValue(0.5)
        self.main_layout.addWidget(self. instrument)
        
        create_psf_widget = QWidget(self)
        self.create_psf_button = QPushButton('Create PSF', self)
        # combo with 'Theoretical PSF', 'PSF Image' and 'Extracted PSF'
        self.combo_box_psf_type = QComboBox(self)
        self.combo_box_psf_type.addItems(["Theoretical PSF", "PSF Image", "Extracted PSF"])
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.create_psf_button)
        h_layout2.addWidget(self.combo_box_psf_type)
        create_psf_widget.setLayout(h_layout2)
        self.main_layout.addWidget(create_psf_widget)
        self.create_psf_button.clicked.connect(self.create_psf_clicked)

        self.open_measured_psf = NapariImageChooserWidget(self.viewer, self, button_text="Choose PSF or Bead Image")
        self.main_layout.addWidget(self.open_measured_psf)

        self.richardson_lucy = RichardsonLucyWidget()
        self.main_layout.addWidget(self.richardson_lucy)
   
        self.setAcceptDrops(True)
        from tnia.gui.widgets.image_batch_test_op import ImageBatchTestOp
        self.main_layout.addWidget(ImageBatch(self, ImageBatchTestOp(self.viewer), self.viewer))        
          
    def open_instrument_models_folder_clicked(self):
        options = QFileDialog.Options()

        # Open a file dialog to select a file
        self.instrument_models_path = QFileDialog.getExistingDirectory(self, 'Open File', '', options=options)
        self.label_instrument_models_folder.setText(self.instrument_models_path)

        json_ = json.load(open(self.instrument_models_path + "/instrument_models.json"))

        self.combo_box_select_psf.clear()
        for model in json_["models"]:
            self.combo_box_select_psf.addItem(model["model_name"])

    def create_psf_clicked(self):
        print("add theoretical psf")

        # check if the instrument models folder is selected
        if self.instrument_models_path == "":
            QMessageBox.information(self, "Warning", "Please select the instrument models folder first.")
            return
        else:

            json_file_name = self.instrument_models_path + "/instrument_models.json"

            xy_spacing = self.instrument.doubleSpinBox_xy_spacing.value()
            z_spacing = self.instrument.doubleSpinBox_z_spacing.value()
            xy_size = self.instrument.spinBox_xy_size.value()
            z_size = self.instrument.spinBox_z_size.value()
            NA = self.instrument.doubleSpinBox_NA.value()
            wavelength = self.instrument.doubleSpinBox_wavelength.value()
            imm_RI=self.instrument.doubleSpinBox_imm_RI.value()
            sample_RI=self.instrument.doubleSpinBox_sample_RI.value()
            depth = self.instrument.doubleSpinBox_depth.value()

            meta_data = {
                InstrumentModelKeys.MODEL_NAME.value: "Theoretical PSF",
                InstrumentModelKeys.FILE.value: "Theoretical PSF",
                InstrumentModelKeys.XY_SPACING.value: xy_spacing,
                InstrumentModelKeys.Z_SPACING.value: z_spacing,
                InstrumentModelKeys.XY_SIZE.value: xy_size,
                InstrumentModelKeys.Z_SIZE.value: z_size,
                InstrumentModelKeys.NA.value: NA,
                InstrumentModelKeys.EMISSION_WAVELENGTH.value: wavelength,
                InstrumentModelKeys.NI.value: imm_RI,
                InstrumentModelKeys.NS.value: sample_RI,
                InstrumentModelKeys.DEPTH.value: depth
            }

            if self.combo_box_psf_type.currentText() == "Theoretical PSF":

                self.psf = gibson_lanni_3D(NA, imm_RI, sample_RI, xy_spacing, z_spacing, xy_size, z_size, depth, wavelength, confocal = False, use_psfm=True)
                self.viewer.add_image(self.psf, name="Theoretical PSF")
            elif self.combo_box_psf_type.currentText() == "PSF Image":
                pass
            elif self.combo_box_psf_type.currentText() == "Extracted PSF":
                # get active napari image
                active_layer = self.viewer.layers.selection.active
                bead_image = active_layer.data

                self.psf, tempi, self.centroids = psf_from_beads(bead_image, background_factor=1.25, apply_median=False, peak_method=1, thresh=0)
                self.viewer.add_image(self.centroids, name="Bead Centroids")
                self.viewer.add_image(self.psf, name="Extracted PSF")
                meta_data[InstrumentModelKeys.MODEL_NAME.value] = "Extracted PSF"
                write_psf(self.instrument_models_path, meta_data, self.psf)

    



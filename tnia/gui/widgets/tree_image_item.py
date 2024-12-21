# imports
from PyQt5.QtWidgets import QTreeWidgetItem
from PyQt5.QtCore import Qt
from tnia.gui.widgets.progress_bar_with_text import ProgressBarWithText

class TreeImageItem(QTreeWidgetItem):
    def __init__(self, parent, text, image):

        super().__init__(parent)

        self.image = image
        self.full_path = image
        self.output = None

        # column 0 is the checkbox 
        self.setCheckState(0, Qt.Unchecked)

        # column 1 is the text 
        self.setText(1, text)

        # column 2 is the progress bar
        self.progress = ProgressBarWithText()
        self.progress.setRange(0, 100)
        self.treeWidget().setItemWidget(self, 2, self.progress)
        
    def update_progress(self, progress):
        self.progress.update_progress(progress)

    def set_out_path(self, out_path):
        self.out_path = out_path


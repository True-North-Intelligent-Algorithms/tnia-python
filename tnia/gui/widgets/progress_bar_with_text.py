from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtGui import QPainter

class ProgressBarWithText(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_text = "Not started"
        
    def setStatusText(self, text):
        self.status_text = text
        self.update()

    def update_progress(self, progress):
        self.setValue(progress)
        self.setStatusText(f" ")
        
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Create a QPainter to draw the text on top of the progress bar
        painter = QPainter(self)
        
        # Calculate the position to draw the text (centered)
        text_width = painter.fontMetrics().width(self.status_text)
        text_height = painter.fontMetrics().height()
        x = (self.width() - text_width) / 2
        y = 2.5*(self.height() - text_height) 
        
        # Set a custom font and color for the text
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        
        # Draw the text
        painter.drawText(x, y, self.status_text)
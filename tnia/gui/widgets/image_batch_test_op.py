# import QThread

from tnia.gui.widgets.image_batch_base_op import ImageBatchBaseOp
from qtpy.QtCore import QThread
from skimage.filters import threshold_otsu

class ImageBatchTestOp(ImageBatchBaseOp):

    def runOp(self, image, update_progress):
        print(image.shape)
        for i in range(100):
            update_progress(i)
            # pause 10 milliseconds
            QThread.msleep(10)

        threshold = threshold_otsu(image)
        return image > threshold

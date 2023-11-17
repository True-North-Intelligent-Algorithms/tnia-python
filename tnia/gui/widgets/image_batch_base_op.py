from skimage import io
import numpy as np
import napari

class ImageBatchBaseOp:
    def __init__(self):
        pass 

    def run(self, image, progress):
        if isinstance(image, str):
            image = io.imread(image)
        elif isinstance(image, napari.layers.image.image.Image):
            image = image.data
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Image should be either a string or a numpy array.")
        
        print("Size of array: ", image.size)
        self.runOp(image, progress)

    def runOp(self, image, progress):
        raise NotImplementedError("This method should be implemented by child classes.")

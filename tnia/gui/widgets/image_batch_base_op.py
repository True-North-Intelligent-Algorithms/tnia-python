from skimage import io
import numpy as np
import napari

class ImageBatchBaseOp:
    def __init__(self, viewer):
        self.viewer = viewer

    def run(self, input, progress):
        if isinstance(input, str):
            image = io.imread(input)
        elif isinstance(input, napari.layers.image.image.Image):
            image = input.data
        elif isinstance(input, np.ndarray):
            image = input
        else:
            raise ValueError("Image should be either a string or a numpy array.")
        
        print("Size of array: ", image.size)
        output = self.runOp(image, progress)

        if isinstance(input, str):
            out_path = input.split(".")[0] + "_out.tif"
            io.imsave(out_path, output)
            return out_path
        elif isinstance(input, napari.layers.image.image.Image):
            self.viewer.add_image(output, name=input.name + "_out")
            return output


    def runOp(self, image, progress):
        raise NotImplementedError("This method should be implemented by child classes.")

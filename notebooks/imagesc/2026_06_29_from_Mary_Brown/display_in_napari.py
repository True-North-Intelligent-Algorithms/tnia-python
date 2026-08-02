import napari
from pathlib import Path
from skimage.io import imread

viewer = napari.Viewer()

parent_path = Path(r'/home/bnorthan/images/tnia-python-images/imagesc/2026_06_27_from_Mary_Brown/')

image_path = parent_path / 'S2C8_159_156_193_DenoisTopHat30.tif'

img = imread(image_path)
viewer.add_image(img, name='S2C8_159_156_193_DenoisTopHat30', colormap='gray')
# viewer.open(image_path)

roi_path = parent_path / 'S2C8_159_156_193_DenoisTopHat30_rois'

from roifile import ImagejRoi

coords_list = []
for roi_file in sorted(roi_path.glob('*.roi')):
    roi = ImagejRoi.fromfile(roi_file)
    coords = roi.integer_coordinates
    # swap x and y (napari is (y, x), roi is (x, y))
    coords_swapped = coords[:, [1, 0]]
    coords_list.append(coords_swapped)

    coords_swapped[:, 0] += roi.top
    coords_swapped[:, 1] += roi.left

viewer.add_shapes(coords_list, shape_type='polygon', name='ROIs',
                  edge_color='red', face_color='transparent')

napari.run()
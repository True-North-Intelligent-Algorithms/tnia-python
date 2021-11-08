import os
from tnia.io.io_helper import get_file_names_from_dir
from skimage.io import imread
from tnia.plotting.projections import show_xyz_max
from tnia.reports.markdown import image_test
report_dir = '../../docs/tests/'

image_dir = 'D:\\images\\ABRF LMRG Image Analysis Study\\nuclei\\'

if os.path.exists(report_dir)==False:
    os.makedirs(report_dir)

# get list files
files = get_file_names_from_dir(image_dir, 'tif')

markdown = ''

for f in files:
    print(f)
    im_name = os.path.basename(f).split('.')[0]
    print(im_name)
    im = imread(f)
    fig = show_xyz_max(im)
    info=['max projection','no processing yet']
    markdown+=image_test(im_name,report_dir, info, fig)

outfile=open(report_dir+'abrf.md', "w")
outfile.write(markdown)
outfile.close()


    

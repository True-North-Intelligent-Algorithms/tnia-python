import os
from tnia.io.io_helper import get_file_names_from_dir
from skimage.io import imread
from tnia.io.tifffile_helper import open_ij3D
from tnia.plotting.projections import show_xyz_max
from tnia.reports.markdown import image_test
from skimage.morphology import white_tophat, cube, ball
from skimage.filters import median, threshold_otsu
from skimage.morphology import binary_opening
from tnia.segmentation.separate import separate_touching2, separate_touching
from skimage.transform import resize
from tnia.viewing.napari_helper import show_image, show_image_and_label
from tnia.deconvolution.psfs import gibson_lanni_3D
from clij2fft.richardson_lucy import richardson_lucy, richardson_lucy_nc
from skimage.morphology import remove_small_holes
from skimage.measure import regionprops

# get the PSF for deconvolution 
x_voxel_size = 0.100
z_voxel_size=0.800

xy_psf_dim=64
z_psf_dim=100

NA=0.75
ni=1
ns=1

psf = gibson_lanni_3D(NA, ni, ns, x_voxel_size, z_voxel_size, xy_psf_dim, z_psf_dim, 0, 0.45)
psf.astype('float32')

report_dir = '../../docs/tests/'

image_dir = 'D:\\images\\ABRF LMRG Image Analysis Study\\nuclei\\'

if os.path.exists(report_dir)==False:
    os.makedirs(report_dir)

# get list files
files = get_file_names_from_dir(image_dir, 'tif')

markdown = ''

#files=[files[3]]
#files=["D:\\images\\ABRF LMRG Image Analysis Study\\nuclei\\nuclei2_out_c90_dr90_image_decon.tif"]

for f in files:
    print(f)
    im_name = os.path.basename(f).split('.')[0]
    print(im_name)

    im_orig, sx, sy, sz = open_ij3D(f)
    im_decon=richardson_lucy_nc(im_orig, psf, 200, 0)
    
    #im = imread(f)
    sx, sy, sz = 0.12, 0.12, 0.2

    im= resize(im_decon, [int(sz*im_orig.shape[0]/sx),im_orig.shape[1], im_orig.shape[2]])
    print(im_orig.shape)

    im_bgs = white_tophat(im, cube(60))
    '''  
    binary = im_bgs>threshold_otsu(im_bgs)

    for i in range(binary.shape[0]):
        binary[i,:,:] = remove_small_holes(binary[i,:,:],1000)

    for i in range(binary.shape[1]):
        binary[:,i,:] = remove_small_holes(binary[:,i,:],1000)

    for i in range(binary.shape[2]):
        binary[:,:,i] = remove_small_holes(binary[:,:,i],1000)
    '''

    #binary = im_bgs>threshold_otsu(im_bgs)
    #binary_,labels,distance = separate_touching(binary, 5,0)
    
    labels,distance,binary = separate_touching2(im_bgs, 5, [15,15,15],[5,5,5])
    
    labels = resize(labels, [im_orig.shape[0],im_orig.shape[1], im_orig.shape[2]], preserve_range=True)
    labels = labels.astype('int32')
    #im_median = median(im_bgs, cube(3))
    #im_segmented = im_median>threshold_otsu(im_median)
    #im_segmented = binary_opening(im_segmented, ball(3))

    im_xyz_max=show_xyz_max(im_orig)
    im_xyz_max.suptitle('Original')
    decon_xyz_max=show_xyz_max(im_decon)
    decon_xyz_max.suptitle('Deconvolved')
    labels_xyz_max=show_xyz_max(labels)
    labels_xyz_max.suptitle('Labels')
    
    figs=[im_xyz_max, decon_xyz_max, labels_xyz_max]

    object_list=regionprops(labels,im_orig);

    info=[str(len(object_list))+' objects found']
    markdown+=image_test(im_name,report_dir, info, figs)

    show=False

    if show:
        viewer=show_image(im_bgs,'image',label=False)
        show_image(distance,'distance',label=False,viewer=viewer)
        show_image(labels,'labels', label=True,viewer=viewer)
        show_image(binary,'binary', label=True,viewer=viewer)

outfile=open(report_dir+'abrf.md', "w")
outfile.write(markdown)
outfile.close()


    

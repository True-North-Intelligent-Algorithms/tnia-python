import matplotlib.pyplot as plt
from tnia.deconvolution import psfs, forward, pad
from tnia.simulation import phantoms
from tnia.plotting.projections import show_xyz_max, show_xy_zy_max
import numpy as np
from clij2fft.richardson_lucy import richardson_lucy, richardson_lucy_nc

xy_size=int(512);
psf_xy_size=int(256);

z_size=int(241);
psf_z_size=int(65);

num_blocks=int(4);
r=30

im_size=[z_size, xy_size, xy_size]
psf_size=[psf_z_size, psf_xy_size, psf_xy_size]

block_size=[z_size, int(xy_size/num_blocks), int(xy_size/num_blocks)];

img = np.zeros(im_size); 

for i in range(num_blocks-1):
    for j in range(num_blocks-1):
        print(i,j)

        img[:,i*block_size[1]+int(block_size[1]/2):i*block_size[1]+int(block_size[1]/2)+block_size[1],j*block_size[2]+int(block_size[2]/2):j*block_size[2]+int(block_size[2]/2)+block_size[2]] = phantoms.sphere3d(block_size,r)

pixel_size = 0.05

psf, _ = psfs.gibson_lanni_3D(1.4, 1.53, 1.4, pixel_size, pixel_size, psf_xy_size, psf_z_size, 0, 0.5)
#psf_xyz_small, _ = psfs.gibson_lanni_3D(1.4, 1.53, 1.4, pixel_size, 256, zv, 0.1,0,0.5)
#plt.imshow(psf_xyz_small[int(full_size[0]/2),:,:])

extended_size = [img.shape[0]+2*int(psf.shape[0]/2), img.shape[1]+2*int(psf.shape[1]/2), img.shape[2]+2*int(psf.shape[2]/2)] 
extended_img = pad.pad(img, extended_size,'constant')
extended_psf = pad.pad(psf, extended_size, 'constant')    

extended_forward = forward.forward(extended_img, extended_psf, 100, 100, True)
forward = pad.unpad(extended_forward, img.shape)

decon = richardson_lucy_nc(forward, psf, 100, 0)

fig=show_xy_zy_max(img)
fig=show_xy_zy_max(extended_forward)
fig=show_xy_zy_max(forward)
fig=show_xy_zy_max(decon)



#import tifffile
#tifffile.imwrite('./spheres.tiff', forward.astype('float32'), imagej=True, metadata={'axes': 'ZYX'})
#tifffile.imwrite('./psf.tiff', psf_xyz_small.astype('float32'), imagej=True, metadata={'axes': 'ZYX'})




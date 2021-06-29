import numpy as np
import bioformats as bf
import tnia.io.bioformats_helper as bfh
import tnia.io.tifffile_helper as tfh
# 4 channels
nc=4
nz=5
# 512 by 512
ny = 512
nx = 512

test=((2**16-1)*np.random.rand(nc, nz, ny, nx)).astype('uint16')
'''
import tifffile
test = np.moveaxis(test,0,1)
tfh.save_zcyx('fromthetiffhelper.tif', test)
#bfh.save_4D('fromthehelper2.tif',test)
'''
import imagej
import xarray as xr

#ij = imagej.init()
ij = imagej.init('sc.fiji:fiji:2.1.1')
data = xr.DataArray(test, dims=('C','z','y','x'))
dataset = dataset = ij.py.to_dataset(data)
dataset = ij.py.to_dataset(data)
ij.io().save(dataset, 'frompyfiji.tif')

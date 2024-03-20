import numpy as np
import bioformats as bf
import tifffile
import tnia.io.bioformats_helper as bfh
import tnia.io.tifffile_helper as tfh
# 4 channels

nz=500
nc=4
# 512 by 512
ny = 2048
nx = 1024

test=((2**16-1)*np.random.rand(nz, nc, ny, nx)).astype('float32')

tifffile.imwrite('from_tifffile.tif', test, imagej=True, metadata={'axes': 'ZCYX', 'mode':'composite'})
tfh.save_zcyx('from_helper.tif', test)

'''
import tifffile
test = np.moveaxis(test,0,1)
tfh.save_zcyx('fromthetiffhelper.tif', test)
#bfh.save_4D('fromthehelper2.tif',test)
'''

'''
import imagej
import xarray as xr

#ij = imagej.init()
ij = imagej.init('sc.fiji:fiji:2.1.1')
data = xr.DataArray(test, dims=('C','z','y','x'))
dataset = dataset = ij.py.to_dataset(data)
dataset = ij.py.to_dataset(data)
ij.io().save(dataset, 'frompyfiji.tif')
'''
from clij2fft.richardsonlucy import getclij2fftlib
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time

# open image and psf
imgName='D:\\images/images/Bars-G10-P15-stack-cropped.tif'
psfName='D:\\images/images/PSF-Bars-stack-cropped.tif'
img=io.imread(imgName)
psf=io.imread(psfName)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img (max projection)')

ax[1].imshow(psf.max(axis=0))
ax[1].set_title('psf (max projection)')

# precondition image and psf
img=img.astype(np.float32)
psf=psf.astype(np.float32)
shifted_psf = np.fft.ifftshift(psf)
result = np.copy(img);
normal=np.ones(img.shape).astype(np.float32)

clij2fft = getclij2fftlib()

# deconvolution using clij2fft
start=time.time()
clij2fft.deconv3d_32f(100, int(img.shape[2]), int(img.shape[1]), int(img.shape[0]), img, shifted_psf, result, normal)
end=time.time()
print('time is',end-start)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(result.max(axis=0))
ax[1].set_title('result')

plt.show()

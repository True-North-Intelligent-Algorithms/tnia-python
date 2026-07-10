from abc import update_abstractmethods
import dis
import numpy as np
import cupy as cp
from cupy import ElementwiseKernel
import timeit
from tnia.deconvolution.pad import pad, unpad, get_next_smooth
from tnia.metrics.errors import RMSE

def fftconv(x, H, shape):
	return cp.fft.irfftn(cp.fft.rfftn(x) * H, s=shape)

nan_to_zero = ElementwiseKernel(
    'float32 x',
    'float32 y',
    '''
    y = (x != x) ? 0.0f : x;
    ''',
    'nan_to_zero'
)

def kldiv(p, q):
  p = p + 1E-4
  q = q + 1E-4
  p = p / cp.sum(p)
  q = q / cp.sum(q)
  kldiv = p * (cp.log(p) - cp.log(q))
  kldiv[cp.isnan(kldiv)] = 0
  #nan_to_zero(kldiv, kldiv) 
  kldiv = cp.sum(kldiv)
  return kldiv

# Define a CUDA kernel for gradient consensus, i.e.
# only update pixels in full estimate where split updates agree in 'sign'
gradient_consensus = ElementwiseKernel(
    'float32 recon, float32 ratio, float32 r1, float32 r2',
    'float32 out',
    '''
    bool skip = (r1 - 1.0f)*(r2 - 1.0f) < 0;
    out = skip ? recon : recon * ratio;
    ''',
    'gradient_consensus'
)

gradient_consensus_max_disagreement = ElementwiseKernel(
    'float32 recon, float32 ratio, float32 disagreement, float32 max_disagreements',
    'float32 out',
    '''
    bool skip = disagreement > max_disagreements;
    out = skip ? recon : recon * ratio;
    ''',
    'gradient_consensus'
)

def rlgc_pt(image, psf_temp, total_iters=-1, auto_stop=True, mask=None, print_details=False, truth=None, noncirc=False, seed=259, thin=True):

  # Load PSF
  #psf_temp = files.upload_file('psf.tif')
  
  # Add new z-axis if we have 2D data
  if image.ndim == 2:
    image = np.expand_dims(image, axis=0)

  if psf_temp.ndim == 2:
    psf_temp = np.expand_dims(psf_temp, axis=0)

  # Pad to size of image
  psf = np.zeros(image.shape)
  psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
  for axis, axis_size in enumerate(psf.shape):
    psf = np.roll(psf, int(axis_size / 2), axis=axis)
  for axis, axis_size in enumerate(psf_temp.shape):
    psf = np.roll(psf, -int(axis_size / 2), axis=axis)
  psf = np.fft.ifftshift(psf)
  psf = psf / np.sum(psf)
  
  # Load data and PSF onto GPU
  image = cp.array(image, dtype=cp.float32)
  psf = cp.array(psf, dtype=cp.float32)

  # Calculate OTF and transpose
  otf = cp.fft.rfftn(psf)
  otfT = cp.conjugate(otf)
  del psf

  # Get dimensions of data
  num_z = image.shape[0]
  num_y = image.shape[1]
  num_x = image.shape[2]
  num_pixels = num_z * num_y * num_x

  # Calculate Richardson-Lucy iterations
  recon = cp.mean(image) * cp.ones((num_z, num_y, num_x), dtype=cp.float32)
  total_disagreements = cp.zeros((num_z, num_y, num_x), dtype=cp.float32)
  previous_recon = recon

  num_iters = 0
  prev_kldim = np.inf
  prev_kld1 = np.inf
  prev_kld2 = np.inf
  start_time = timeit.default_timer()

  rng = cp.random.default_rng(seed)

  #while True:
  for i in range(total_iters):
    iter_start_time = timeit.default_timer()

    split_start = timeit.default_timer()
    # Split recorded image into 50:50 images
    if thin == True:
      split = rng.binomial(image.astype('int64'), p=0.5)
      factor = 0.5
    else:
      split = image
      factor = 1.0
    
    print("Split time: %1.5f s." % (timeit.default_timer() - split_start))

    conv_start = timeit.default_timer()
    # Calculate prediction
    Hu = fftconv(recon, otf, image.shape)
    print("Convolution time: %1.5f s." % (timeit.default_timer() - conv_start))

    # Calculate KL divergences and stop iterations if both have increased
    hratio_start = timeit.default_timer()
    # Calculate updates for split images and full images (H^T (d / Hu))
    HTratio = fftconv(cp.divide(split, factor * (Hu + 1E-12), dtype=cp.float32), otfT, image.shape)
    del split
    print("H^T ratio time: %1.5f s." % (timeit.default_timer() - hratio_start))

    # Save previous estimate in case KLDs increase after this iteration
    previous_recon = recon

    
    update_start= timeit.default_timer()
    # Update estimate
    #recon = gradient_consensus(recon, HTratio, HTratio1, HTratio2)
    recon = recon*HTratio
    num_iters = num_iters + 1

  return recon.get()

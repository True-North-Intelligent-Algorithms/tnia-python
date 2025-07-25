from abc import update_abstractmethods
import dis
from tracemalloc import stop
import numpy as np
import cupy as cp
from cupy import ElementwiseKernel
import timeit
from tnia.deconvolution.pad import pad, unpad, get_next_smooth
from errors import RMSE

def fftconv(x, H, shape):
	return cp.fft.irfftn(cp.fft.rfftn(x) * H, s=shape)

def kldiv(p, q):
	p = p + 1E-4
	q = q + 1E-4
	p = p / cp.sum(p)
	q = q / cp.sum(q)
	kldiv = p * (cp.log(p) - cp.log(q))
	kldiv[cp.isnan(kldiv)] = 0
	kldiv = cp.sum(kldiv)
	return kldiv

# Define a CUDA kernel for application of consensus map
filter_update = ElementwiseKernel(
    'float32 recon, float32 HTratio, float32 consensus_map',
    'float32 out',
    '''
    bool skip = consensus_map < 0;
    out = skip ? recon : recon * HTratio
    ''',
    'filter_update'
)


def rlgc_latest(image, psf_temp, total_iters=-1, auto_stop=True, truth=None, seed=42):
  
  rng = cp.random.default_rng(seed)  
    
  stats = {}
  stats['iteration'] = []
  stats['rmse'] = []
  stats['kldim'] = []
  stats['kld1'] = []
  stats['kld2'] = []

  # Load PSF
  #psf_temp = files.upload_file('psf.tif')
  
  # Add new z-axis if we have 2D data
  if image.ndim == 2:
    image = np.expand_dims(image, axis=0)

  if psf_temp.ndim == 2:
    psf_temp = np.expand_dims(psf_temp, axis=0)
  
  if truth is not None:
    if truth.ndim == 2:
      truth = np.expand_dims(truth, axis=0)

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
  truth = cp.array(truth, dtype=cp.float32) if truth is not None else None

  # Calculate OTF and transpose
  # Calculate OTF and transpose
  otf = cp.fft.rfftn(psf)
  otfT = cp.conjugate(otf)
  otfotfT = otf * otfT
  del psf

  # Get dimensions of data
  num_z = image.shape[0]
  num_y = image.shape[1]
  num_x = image.shape[2]
  num_pixels = num_z * num_y * num_x

  # Calculate Richardson-Lucy iterations
  #recon = cp.mean(image) * cp.ones((num_z, num_y, num_x), dtype=cp.float32)
  recon = cp.mean(image) * cp.ones_like(image)
  previous_recon = recon

  num_iters = 0
  prev_kldim = np.inf
  prev_kld1 = np.inf
  prev_kld2 = np.inf
  start_time = timeit.default_timer()

  stop_iteration = -1

  #while True:
  for i in range(total_iters):
    iter_start_time = timeit.default_timer()

    stats['iteration'].append(i)
    
    if truth is not None:
        print(recon.mean(), truth.mean())
        rmse = RMSE(recon, truth)#, backend=cp)
        stats['rmse'].append(rmse.get())
    
    # Split recorded image into 50:50 images
    split1 = rng.binomial(image.astype('int64'), p=0.5)
    split2 = image - split1

    # Calculate prediction
    Hu = fftconv(recon, otf, image.shape)

    # Calculate KL divergences and stop iterations if both have increased
    kldiv_start = timeit.default_timer()
    kldim = kldiv(Hu, image)
    kld1 = kldiv(Hu, split1) 
    kld2 = kldiv(Hu, split2)

    stats['kldim'].append(kldim)
    stats['kld1'].append(kld1)
    stats['kld2'].append(kld2)
    
    if ((kld1 > prev_kld1) & (kld2 > prev_kld2)):
           
        if stop_iteration == -1:
            stop_iteration = i
            best_recon = previous_recon
        
            print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
        
        if auto_stop:
            return best_recon.get(), best_recon.get(), stats, stop_iteration
    
    del previous_recon
    prev_kldim = kldim
    prev_kld1 = kld1
    prev_kld2 = kld2

    # Calculate updates for split images and full images (H^T (d / Hu))
    HTratio1 = fftconv(cp.divide(split1, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image.shape)
    del split1
    HTratio2 = fftconv(cp.divide(split2, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image.shape)
    del split2
    HTratio = (HTratio1 + HTratio2)/2
    del Hu

    # Save previous estimate in case KLDs increase after this iteration
    previous_recon = recon    
    
    # Calculate gradient consensus and blur with the interaction kernel, i.e. H^T H
    consensus_map = fftconv((HTratio1 - 1) * (HTratio2 - 1), otfotfT, recon.shape)

    # Update our reconstruction only where the blurred consensus map says we should
    recon = filter_update(recon, HTratio, consensus_map)
    
    # Calculate update statistics    
    min_HTratio = cp.min(HTratio)
    max_HTratio = cp.max(HTratio)
    max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
    del HTratio
    
    calc_time = timeit.default_timer() - iter_start_time
    print("Iteration %03d completed in %1.3f s. KLDs = %1.4f (image), %1.4f (split 1), %1.4f (split 2). Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, kldim, kld1, kld2, min_HTratio, max_HTratio, max_relative_delta))

    num_iters = num_iters + 1
    
  recon = recon.get()
  if stop_iteration > 0:
     best_recon = best_recon.get()
  else:
     best_recon = recon

  return recon, best_recon, stats, stop_iteration

import numpy as np
import cupy as cp
import timeit

# This code is written by James Manton and the original can be found here https://github.com/jdmanton/gpu_rl_deconv
# It's copied here for convenience and to make it easier to compare the performance of the different deconvolution implementations
# This file is an almost exact copy of the original and is property of the original author 

def fftconv(x, H):
	return cp.fft.irfftn(cp.fft.rfftn(x) * H, x.shape)

def fftdeconv(image, psf_temp, num_iters=100, process_psf=False, reblurred=None):

    if (process_psf):
        print("Processing PSF...")
        # Take upper left 16x16 pixels to estimate noise level and create appropriate fake noise
        noisy_region = psf_temp[0:16, 0:16, 0:16]
        psf = np.random.normal(np.mean(noisy_region), np.std(noisy_region), image.shape)
    else:
        psf = np.zeros(image.shape)

    psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
    for axis, axis_size in enumerate(psf_temp.shape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    if (process_psf):	
        psf = psf - np.mean(noisy_region)
        psf[psf < 0] = 0

    # Load data and PSF onto GPU
    image = cp.array(image, dtype=cp.float32)
    psf = cp.array(psf, dtype=cp.float32)

    # Calculate OTF and transpose
    otf = cp.fft.rfftn(psf)
    psfT = cp.flip(psf, (0, 1, 2))
    otfT = cp.fft.rfftn(psfT)

    # Log which files we're working with and the number of iterations
    print('Input shape: %s' % (image.shape, ))
    print('PSF shape: %s' % (psf_temp.shape, ))
    print('Number of iterations: %d' % num_iters)
    print('PSF processing: %s' % process_psf)
    print('')

    # Get dimensions of data
    num_z = image.shape[0]
    num_x = image.shape[1]
    num_y = image.shape[2]

    # Calculate Richardson-Lucy iterations
    HTones = fftconv(cp.ones_like(image), otfT)
    recon = cp.ones((num_z, num_x, num_y))

    for iter in range(num_iters):
        start_time = timeit.default_timer()
        Hu = fftconv(recon, otf)
        ratio = image / (Hu + 1E-12)
        HTratio = fftconv(ratio, otfT)
        recon = recon * HTratio / HTones
        calc_time = timeit.default_timer() - start_time

        # if iter % 10 == 0:
        if iter % 10 == 0:
            print(iter, end =" ")
        #print("Iteration %d completed in %f s." % (iter + 1, calc_time))

    # Reblur, collect from GPU and save if argument given
    if reblurred is not None:
        reblurred = fftconv(recon, otf)
        reblurred = reblurred.get()

    recon = recon.get()

    if reblurred is not None:
        return recon, reblurred
    else:
        return recon
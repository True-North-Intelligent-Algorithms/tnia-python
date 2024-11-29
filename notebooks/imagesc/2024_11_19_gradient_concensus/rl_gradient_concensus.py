import numpy as np
import cupy as cp
import os
import timeit

seed = 363
rng = np.random.default_rng(seed)

def pad_psf(psf_temp, image):

    # Add new z-axis if we have 2D data
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

    return psf

def fftconv(x, H):
	return cp.real(cp.fft.ifftn(cp.fft.fftn(x) * H))

def kldiv(p, q):
	p = p + 1E-4
	q = q + 1E-4
	p = p / cp.sum(p)
	q = q / cp.sum(q)
	kldiv = p * (cp.log(p) - cp.log(q))
	kldiv[cp.isnan(kldiv)] = 0
	kldiv = cp.sum(kldiv)
	return kldiv

def rlgc(image, psf, total_iters, auto_stop=True, mask=None):

    # Calculate OTF and transpose
    otf = cp.fft.fftn(psf)
    otfT = cp.conjugate(otf)

    # Get dimensions of data
    num_z = image.shape[0]
    num_y = image.shape[1]
    num_x = image.shape[2]
    num_pixels = num_z * num_y * num_x

    HTones = cp.ones_like(image)
    if mask is not None:
         HTones = HTones * mask
         image = image * mask
    # Calculate Richardson-Lucy iterations
    HTones = fftconv(HTones, otfT)
    
    
    recon = cp.mean(image) * cp.ones((num_z, num_y, num_x), dtype=cp.float32)
    previous_recon = recon

    num_iters = 0
    prev_kldim = np.inf
    prev_kld1 = np.inf
    prev_kld2 = np.inf
    start_time = timeit.default_timer()

    #while True:
    for i in range(total_iters):
        iter_start_time = timeit.default_timer()

        # Split recorded image into 50:50 images
        # TODO: make this work on the GPU (for some reason, we get repeating blocks with a naive conversion to cupy)
        split1 = rng.binomial(image.get().astype('int64'), p=0.5)
        split1 = cp.array(split1)
        split2 = image - split1

        # Calculate prediction
        Hu = fftconv(recon, otf)

        # Calculate KL divergences and stop iterations if both have increased
        mask_val = 100
        print(f"Hu min: {cp.min(Hu)}, max: {cp.max(Hu)}") 
        kldim = kldiv(Hu, image)
        kld1 = kldiv(Hu, split1)
        kld2 = kldiv(Hu, split2)
        if ((kld1 > prev_kld1) & (kld2 > prev_kld2)):
            recon = previous_recon
            print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
            if auto_stop:
                return recon 
        
        del previous_recon
        prev_kldim = kldim
        prev_kld1 = kld1
        prev_kld2 = kld2

        # Calculate updates for split images and full images (H^T (d / Hu))
        HTratio1 = fftconv(split1 / (0.5 * (Hu + 1E-12)), otfT) / HTones
        del split1
        HTratio2 = fftconv(split2 / (0.5 * (Hu + 1E-12)), otfT) / HTones
        del split2
        HTratio = fftconv(image / (Hu + 1E-12), otfT) / HTones
        del Hu

        # Normalise update steps by H^T(1) and only update pixels in full estimate where split updates agree in 'sign'
        shouldNotUpdate = (HTratio1 - 1) * (HTratio2 - 1) < 0
        del HTratio1
        del HTratio2
        HTratio[shouldNotUpdate] = 1
        num_updated = num_pixels - cp.sum(shouldNotUpdate)
        del shouldNotUpdate

        # Save previous estimate in case KLDs increase after this iteration
        previous_recon = recon

        # Update estimate
        recon = recon * HTratio
        min_HTratio = cp.min(HTratio)
        max_HTratio = cp.max(HTratio)
        max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
        del HTratio

        calc_time = timeit.default_timer() - iter_start_time
        print("Iteration %03d completed in %1.3f s. KLDs = %1.4f (image), %1.4f (split 1), %1.4f (split 2). %1.2f %% of image updated. Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, kldim, kld1, kld2, 100 * num_updated / num_pixels, min_HTratio, max_HTratio, max_relative_delta))

        num_iters = num_iters + 1
    
    if (mask is not None):
        recon = recon*mask
        recon[ (1-mask) == 1] = recon.max() #recon*mask + mask_values
    
    return recon


import numpy as np
import cupy as cp
import timeit
from tnia.deconvolution.pad import pad, unpad, get_next_smooth
from tnia.metrics.errors import RMSE

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

def rlgc(image, psf, total_iters, auto_stop=True, mask=None, print_details=False, truth=None, noncirc=False, seed=259):
    """
    Vendored from:
    
    https://colab.research.google.com/drive/1mfVNSCaYHz1g56g92xBkIoa8190XNJpJ

    Below is the original description of the algorithm from the colab notebook:

    Richardson–Lucy deconvolution iteratively updates an estimate of an object 
    in a way that maximizes the likelihood that, when blurred, the measured 
    image data is produced. However, as the image data contains noise, letting 
    the algorithm continue for too many iterations leads to artifacts as the 
    object is overfit to the noise, not the true signal. This problem is 
    normally attacked via explicit regularization, in which a penalty term is 
    added to the likelihood function to promote a more 'reasonable' solution. 
    However, while the Richardson–Lucy algorithm is physically justified by 
    the statistics of shot noise, choosing a 'reasonable' regularization is 
    not. In addition to selecting an appropriate regularization function, the 
    weight of this regularization must also be tuned by hand.

    As an alternative approach, we have developed a form of implicit 
    regularization, which we call gradient consensus. This augments the usual 
    Richardson–Lucy iteration with a step in which two independent images are 
    generated from the original measured image, via Poisson thinning. This is 
    akin to placing a 50:50 beamsplitter and second detector in front of the 
    original detector. For every photon incident on the beamsplitter flip a 
    coin — if heads, the photon goes to detector 1 and, if tails, the photon 
    goes to detector 2. Both of these images contain the same true signal, but 
    now have independent noise realizations.

    Importantly, both images are still Poisson-noisy and therefore suitable 
    for input into the Richardson–Lucy algorithm. As such, the current 
    estimate is used to compute an update step for each image. Where both 
    updates agree about whether a pixel value in the estimate should be made 
    larger (smaller), we allow the update calculated for the original image 
    (which contains all the photons) to be applied to the estimate. Where 
    there is not a gradient consensus, the update is not applied.

    Eventually, any updates made to the estimate arise only because of random 
    redistributions of photons into the split images and so we consider the 
    true signal to be well-fit and terminate the algorithm. In addition, as 
    this stopping criterion operates locally, high SNR regions of the image 
    are allowed to iterate further than low SNR regions, ensuring that all 
    parts of the image are deconvolved optimally.

    Additional:

    In addition I've added the following
    
    1.  Make HTones array a map of regions where good pixels were aquired.  This approach handles both 
    bad pixels and edges by essentially treating the forward model as having pixels with 0 intensity in these locations. 
    Saturated pixels need to be indicated by passing in a mask.  Edge pixels are handled by extending the 
    image by zeros essentially making the reconstructed space larger than the image window.

    2.  Adding the option to over-ride the stop criteria and run for an exact number of iterations. 

    3.  Return some statistics of error at each iteration (if simulation and a ground truth is provided).

    4. The 'should not update' calculation (HRatio[shouldNotUpdate] = 1) is done with a cuda kernel to improve speed.  

    5.  Make seed a parameter to allow for testing both reproducibility (how similar results are with different seeds) and 
    repeatability (whether the same seed gives the same result).

    """

    mempool = cp.get_default_memory_pool()


    total_gpu_memory = mempool

    bpg=(1024**3)

    available_gpu_memory = cp.cuda.Device(0).mem_info[0]
    total_gpu_memory = cp.cuda.Device(0).mem_info[1]
    print("Total GPU memory = {}".format(total_gpu_memory/bpg))
    print("Available GPU memory = {}".format(available_gpu_memory/bpg))
    print("At beginning, used = {}".format(mempool.used_bytes()/bpg))

    rng_cp = cp.random.default_rng(seed)
    
    stats = {}
    stats['iteration'] = []
    stats['rmse'] = []
    stats['kldim'] = []
    stats['kld1'] = []
    stats['kld2'] = []

    if mask is None:
        mask = np.ones_like(image)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
        
        if mask is not None:        
            mask = np.expand_dims(mask, axis=0)

        if truth is not None:
            truth = np.expand_dims(truth, axis=0)

    if psf.ndim == 2:
        psf = np.expand_dims(psf, axis=0)

    # make image and psf cupy arrays
    image = cp.array(image, dtype=cp.float32)
    psf = cp.array(psf, dtype=cp.float32)

    if mask is not None:
        mask = cp.array(mask, dtype=cp.float32)

    if truth is not None:
        truth = cp.array(truth, dtype=cp.float32)
    
    # Get dimensions of data
    num_z = image.shape[0]
    num_y = image.shape[1]
    num_x = image.shape[2]
    num_pixels = num_z * num_y * num_x
    
    print("after truth and mask used = {}".format(mempool.used_bytes()/bpg))

    HTones = cp.ones_like(image)
    if mask is not None:
         HTones = HTones * mask
         image = image * mask
    
    print("after HTones used = {}".format(mempool.used_bytes()/bpg))

    # BN if noncirc==True then pad the image, psf and HTOnes array
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]
        
        # get next fast FFT size
        extended_size = get_next_smooth(extended_size)

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')
        psf = cp.fft.ifftshift(psf)

        if truth is not None:
            truth,_=pad(truth, extended_size, 'constant')
        if mask is not None:
            mask,_=pad(mask, extended_size, 'constant')
    else:
        #psf = pad_psf(psf, image)
        psf,_=pad(psf, image.shape, 'constant')
        psf = cp.fft.ifftshift(psf)
    
    # Calculate OTF and transpose
    otf = cp.fft.fftn(psf)
    otfT = cp.conjugate(otf)
    
    # Calculate Richardson-Lucy iterations
    delta = 1e-6
    HTones_otfT = fftconv(HTones, otfT)
    HTones_otfT[HTones_otfT < delta] = 1

    recon = cp.mean(image) * cp.ones_like(image)
    previous_recon = recon

    num_iters = 0
    prev_kldim = np.inf
    prev_kld1 = np.inf
    prev_kld2 = np.inf
    start_time = timeit.default_timer()

    # BN the shouldNotUpdate array is used to determine which pixels should not be updated
    # Using conditional indexing to set the pixels 'not to update' by HTRatio[shouldNotUpdate] = 1 is slow
    # so do it with an elementwise kernel instead
    elemntwise_should_not_update = cp.ElementwiseKernel(
        'S x',
        'T y',
        'y = (x == 1)?1:y',
        'fast_index')
    
    stopped = False
    stop_iteration = -1
    
    #while True:
    for i in range(total_iters):

        if i % 10 == 0:
            print(i, end =" ")
        
        stats['iteration'].append(i)
        
        if truth is not None:
            rmse = RMSE(recon*HTones, truth, HTones)
            stats['rmse'].append(rmse.get())
        
        iter_start_time = timeit.default_timer()

        # Split recorded image into 50:50 images
        # TODO: make this work on the GPU (for some reason, we get repeating blocks with a naive conversion to cupy)
        cprand = True
        if not cprand:
            split1 = rng.binomial(image.get().astype('int64'), p=0.5)
            split1 = cp.array(split1)
        else:
            split1 = rng_cp.binomial(image.astype('int64'), p=0.5)

        split2 = image - split1

        # Calculate prediction
        Hu = fftconv(recon, otf)

        # Calculate KL divergences and stop iterations if both have increased
        # BN Since HTOnes is a map of regions where good pixels were acquired (bad pixels such as saturated pixels and out of bounds pixels are set to 0),
        # only use the pixels where HTOnes is not 0 to calculate the KL divergence 
        kldim = kldiv(Hu*HTones, image*HTones)
        kld1 = kldiv(Hu*HTones, split1*HTones)
        kld2 = kldiv(Hu*HTones, split2*HTones)

        stats['kldim'].append(kldim)
        stats['kld1'].append(kld1)
        stats['kld2'].append(kld2)
        
        if ((kld1 > prev_kld1) & (kld2 > prev_kld2)):
            
            stopped = True
            
            if stop_iteration == -1:
                stop_iteration = i
            
            #print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
            if auto_stop:
                recon = previous_recon
                return recon 
        
        prev_kldim = kldim
        prev_kld1 = kld1
        prev_kld2 = kld2

        del previous_recon

        # Calculate updates for split images and full images (H^T (d / Hu))
        HTratio1 = fftconv(split1 / (0.5 * (Hu + 1E-12)), otfT) / HTones_otfT
        del split1
        HTratio2 = fftconv(split2 / (0.5 * (Hu + 1E-12)), otfT) / HTones_otfT
        del split2
        HTratio = fftconv(image / (Hu + 1E-12), otfT) / HTones_otfT
        del Hu

        # Normalise update steps by H^T(1) and only update pixels in full estimate where split updates agree in 'sign'
        shouldNotUpdate = (HTratio1 - 1) * (HTratio2 - 1) < 0
        del HTratio1
        del HTratio2
        
        #HTratio[shouldNotUpdate] = 1
        elemntwise_should_not_update(shouldNotUpdate, HTratio)
        
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

        if print_details:
            if auto_stop:
                print("Iteration %03d completed in %1.3f s. KLDs = %1.4f (image), %1.4f (split 1), %1.4f (split 2). %1.2f %% of image updated. Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, kldim, kld1, kld2, 100 * num_updated / num_pixels, min_HTratio, max_HTratio, max_relative_delta))
            else:
                print("Iteration %03d completed in %1.3f s. %1.2f %% of image updated. Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, 100 * num_updated / num_pixels, min_HTratio, max_HTratio, max_relative_delta))
        
            print()
        num_iters = num_iters + 1

    print()  
    
    if (mask is not None):
        recon = recon*mask
        recon[ (1-mask) == 1] = recon.max() #recon*mask + mask_values
    
    recon = recon.get()
    
    if noncirc:
        recon = unpad(recon, original_size)
    
    recon = np.squeeze(recon)
    
    return recon, stats, stop_iteration



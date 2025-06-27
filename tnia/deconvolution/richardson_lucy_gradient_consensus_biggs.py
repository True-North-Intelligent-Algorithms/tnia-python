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

# Define a CUDA kernel for gradient consensus, i.e.
# only update pixels in full estimate where split updates agree in 'sign'
# (also clamp Biggs-Andrews update to zero in case it goes too far)
gradient_consensus = ElementwiseKernel(
    'float32 recon, float32 ratio, float32 r1, float32 r2',
    'float32 out',
    '''
    bool skip = (r1 - 1.0f)*(r2 - 1.0f) < 0;
    out = skip ? recon : recon * ratio;
    out = out < 0 ? 0 : out;
    ''',
    'gradient_consensus'
)

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
previous_recon = recon

num_iters = 0
prev_kldim = np.inf
prev_kld1 = np.inf
prev_kld2 = np.inf
start_time = timeit.default_timer()

# Biggs-Andrews acceleration
g2 = cp.empty_like(recon)
g1 = cp.empty_like(recon)

while True:
  iter_start_time = timeit.default_timer()

  # Split recorded image into 50:50 images
  split1 = rng.binomial(image.astype('int64'), p=0.5)
  split2 = image - split1

  # Biggs-Andrews acceleration
  alpha = cp.sum(g1 * g2) / cp.sum(g2 * g2)
  if alpha > 1:
    alpha = 1
  if alpha < 0:
    alpha = 0
  if cp.isnan(alpha):
    alpha = 0
  y = recon + alpha * (recon - previous_recon)

  # Calculate prediction
  Hu = fftconv(y, otf, image.shape)

  # Calculate KL divergences and stop iterations if both have increased
  kldim = kldiv(Hu, image)
  kld1 = kldiv(Hu, split1)
  kld2 = kldiv(Hu, split2)
  if play_it_safe:
    if ((kld1 > prev_kld1) | (kld2 > prev_kld2)):
      recon = previous_recon
      print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
      break
  else:
    if ((kld1 > prev_kld1) & (kld2 > prev_kld2)):
      recon = previous_recon
      print("Optimum result obtained after %d iterations with a total time of %1.1f seconds." % (num_iters - 1, timeit.default_timer() - start_time))
      break
  prev_kldim = kldim
  prev_kld1 = kld1
  prev_kld2 = kld2

  # Biggs-Andrews acceleration
  previous_recon = recon
  recon = y

  # Calculate updates for split images and full images (H^T (d / Hu))
  HTratio1 = fftconv(cp.divide(split1, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image.shape)
  del split1
  HTratio2 = fftconv(cp.divide(split2, 0.5 * (Hu + 1E-12), dtype=cp.float32), otfT, image.shape)
  del split2
  HTratio = fftconv(image / (Hu + 1E-12), otfT, recon.shape)
  del Hu

  # Update estimate
  recon = gradient_consensus(recon, HTratio, HTratio1, HTratio2)

  # Biggs-Andrews acceleration
  g2 = g1
  g1 = recon - y
  del y

  min_HTratio = cp.min(HTratio)
  max_HTratio = cp.max(HTratio)
  max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
  del HTratio

  calc_time = timeit.default_timer() - iter_start_time
  print("Iteration %03d completed in %1.3f s. KLDs = %1.4f (image), %1.4f (split 1), %1.4f (split 2). Update range: %1.2f to %1.2f. Largest relative delta = %1.5f." % (num_iters + 1, calc_time, kldim, kld1, kld2, min_HTratio, max_HTratio, max_relative_delta))


  num_iters = num_iters + 1

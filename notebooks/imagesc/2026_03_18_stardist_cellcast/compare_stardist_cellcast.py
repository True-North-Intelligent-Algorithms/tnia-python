"""
Compare StarDist and CellCast segmentation performance
"""
import numpy as np
from skimage.io import imread, imsave
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import cellcast.models.stardist_2d as sd
import time
import os

# Input image path
im_path = '/home/bnorthan/images/tnia-python-images/imagesc/2026_03_18_stardist_cellcast/s_aureus_stat.tif'
im_dir = os.path.dirname(im_path)
im_filename = os.path.basename(im_path)

# Load image
print("Loading image...")
im = imread(im_path)
print(f"Image: {im_filename}")
print(f"Shape: {im.shape}\n")

# Load models and warmup
print("Loading StarDist model...")
model = StarDist2D.from_pretrained('2D_versatile_fluo')

print("Warming up models...")
im_normalized = normalize(im, 1, 99.8)
_ = model.predict_instances(im_normalized, prob_thresh=0.5, nms_thresh=0.3)
_ = sd.predict_versatile_fluo(im, gpu=True)
print("Warmup complete\n")

# Run StarDist
print("Running StarDist...")
start = time.time()
im_normalized = normalize(im, 1, 99.8)
labels_sd, _ = model.predict_instances(im_normalized, prob_thresh=0.5, nms_thresh=0.3)
time_sd = time.time() - start

# Run CellCast
print("Running CellCast...")
start = time.time()
labels_cc = sd.predict_versatile_fluo(im, prob_threshold=0.5, nms_threshold=0.3, gpu=True)
time_cc = time.time() - start

# Compare results
different_pixels = np.sum(labels_sd != labels_cc)
total_pixels = labels_sd.size
percent_diff = (different_pixels / total_pixels) * 100

# Report
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"\nTiming:")
print(f"  StarDist:  {time_sd:.3f} seconds")
print(f"  CellCast:  {time_cc:.3f} seconds")
print(f"  Speedup:   {time_sd/time_cc:.2f}x")

print(f"\nObjects detected:")
print(f"  StarDist:  {labels_sd.max()}")
print(f"  CellCast:  {labels_cc.max()}")
print(f"  Difference: {abs(labels_sd.max() - labels_cc.max())}")

print(f"\nPixel differences:")
print(f"  Total pixels:      {total_pixels:,}")
print(f"  Different pixels:  {different_pixels:,}")
print(f"  Percent different: {percent_diff:.2f}%")
print("="*50)

# Create comparison visualization
print("\nCreating comparison image...")

# Create binary masks (0 = background, 1 = object)
mask_sd = (labels_sd > 0).astype(np.uint8)
mask_cc = (labels_cc > 0).astype(np.uint8)

# Create RGB comparison image
comparison = np.zeros((*im.shape, 3), dtype=np.uint8)

# Both background (0,0) -> Black [0,0,0]
# StarDist only (1,0) -> Red [255,0,0]
# CellCast only (0,1) -> Green [0,255,0]
# Both detected (1,1) -> Yellow [255,255,0]

both_bg = (mask_sd == 0) & (mask_cc == 0)
sd_only = (mask_sd == 1) & (mask_cc == 0)
cc_only = (mask_sd == 0) & (mask_cc == 1)
both_fg = (mask_sd == 1) & (mask_cc == 1)

comparison[both_bg] = [0, 0, 0]      # Black
comparison[sd_only] = [255, 0, 0]    # Red
comparison[cc_only] = [0, 255, 0]    # Green
comparison[both_fg] = [255, 255, 0]  # Yellow

# Count pixels in each category
n_both_bg = np.sum(both_bg)
n_sd_only = np.sum(sd_only)
n_cc_only = np.sum(cc_only)
n_both_fg = np.sum(both_fg)

print(f"\nPixel categories:")
print(f"  Both background (black):  {n_both_bg:,} ({n_both_bg/total_pixels*100:.2f}%)")
print(f"  StarDist only (red):      {n_sd_only:,} ({n_sd_only/total_pixels*100:.2f}%)")
print(f"  CellCast only (green):    {n_cc_only:,} ({n_cc_only/total_pixels*100:.2f}%)")
print(f"  Both agree foreground (yellow):      {n_both_fg:,} ({n_both_fg/total_pixels*100:.2f}%)")
print(f" Both methods agree (black + yellow): {n_both_bg + n_both_fg:,} ({(n_both_bg + n_both_fg)/total_pixels*100:.2f}%)")

# Save comparison image to same directory as input
output_filename = f"comparison_{os.path.splitext(im_filename)[0]}.tif"
output_path = os.path.join(im_dir, output_filename)
imsave(output_path, comparison)
print(f"\nComparison image saved to: {output_path}")
print("  Black = both background")
print("  Red = StarDist only")
print("  Green = CellCast only")
print("  Yellow = both methods agree foreground")

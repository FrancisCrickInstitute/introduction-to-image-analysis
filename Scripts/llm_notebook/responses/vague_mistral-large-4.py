import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, segmentation, measure, feature, exposure
from scipy import ndimage as ndi

# ==================== CONFIGURATION ====================
# Adjust these parameters based on your image:
CHANNEL = 0              # Which channel to use (0-indexed). Usually the nuclear stain (DAPI/Hoechst)
MIN_CELL_SIZE = 80       # Minimum cell area in pixels (removes noise/debris)
MAX_CELL_SIZE = 10000    # Maximum cell area (removes large clumps)
MIN_DISTANCE = 12        # Minimum pixels between cell centers (for separating touching cells)
SIGMA_SMOOTH = 1.5       # Gaussian smoothing sigma (reduces noise before thresholding)
# =======================================================

# 1. Handle multichannel image and extract the channel to analyze
if img.ndim == 3:
    # Auto-detect if channels are first (C,H,W) or last (H,W,C)
    if img.shape[0] <= 5 and img.shape[0] < img.shape[1]:
        # Likely (C, H, W) format
        img_channel = img[CHANNEL]
        print(f"Detected shape (C,H,W): {img.shape}. Using channel {CHANNEL}")
    else:
        # Likely (H, W, C) format
        img_channel = img[..., CHANNEL]
        print(f"Detected shape (H,W,C): {img.shape}. Using channel {CHANNEL}")
else:
    img_channel = img
    print(f"Single channel image: {img.shape}")

# Convert to float and normalize to 0-1 range for processing
img_float = img_channel.astype(np.float32)
if img_float.max() > 0:
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min())

# 2. Preprocessing
# Background subtraction using a large Gaussian (approximates rolling-ball)
background = filters.gaussian(img_float, sigma=50, preserve_range=True)
img_bg = img_float - background
img_bg[img_bg < 0] = 0

# Smooth to reduce noise
img_smooth = filters.gaussian(img_bg, sigma=SIGMA_SMOOTH, preserve_range=True)

# Contrast stretching for better thresholding
p2, p98 = np.percentile(img_smooth, (2, 98))
img_rescaled = exposure.rescale_intensity(img_smooth, in_range=(p2, p98))

# 3. Thresholding (Otsu's automatic method)
threshold = filters.threshold_otsu(img_rescaled)
binary_mask = img_rescaled > threshold

# 4. Clean the binary mask
# Remove small objects (noise)
binary_clean = morphology.remove_small_objects(binary_mask, min_size=MIN_CELL_SIZE)
# Fill holes inside cells (if any)
binary_clean = morphology.remove_small_holes(binary_clean, area_threshold=MIN_CELL_SIZE)
# Smooth boundaries with morphological opening
binary_clean = morphology.binary_opening(binary_clean, morphology.disk(2))

# 5. Separate touching cells using Watershed algorithm
# Distance transform: distance of each foreground pixel to the background
distance = ndi.distance_transform_edt(binary_clean)

# Find local maxima as seeds for the watershed
coords = feature.peak_local_max(distance, min_distance=MIN_DISTANCE, labels=binary_clean)
seeds = np.zeros(distance.shape, dtype=bool)
seeds[tuple(coords.T)] = True
markers, num_seeds = ndi.label(seeds)

# Apply watershed to separate touching objects
labels = segmentation.watershed(-distance, markers, mask=binary_clean)

# 6. Filter by size and count
regions = measure.regionprops(labels)
valid_cells = [r for r in regions if MIN_CELL_SIZE <= r.area <= MAX_CELL_SIZE]
cell_count = len(valid_cells)

print(f"\n{'='*50}")
print(f"RESULTS:")
print(f"  Threshold value: {threshold:.4f}")
print(f"  Seeds found: {num_seeds}")
print(f"  Valid cells counted: {cell_count}")
print(f"  Rejected regions (too small/large): {len(regions) - cell_count}")
print(f"{'='*50}")

# 7. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Original channel
axes[0, 0].imshow(img_channel, cmap='gray')
axes[0, 0].set_title(f'Original (Channel {CHANNEL})')
axes[0, 0].axis('off')

# Preprocessed
axes[0, 1].imshow(img_smooth, cmap='gray')
axes[0, 1].set_title('Background subtracted & smoothed')
axes[0, 1].axis('off')

# Binary mask
axes[0, 2].imshow(binary_clean, cmap='gray')
axes[0, 2].set_title(f'Binary mask (Otsu threshold)')
axes[0, 2].axis('off')

# Distance transform
axes[1, 0].imshow(distance, cmap='viridis')
axes[1, 0].set_title('Distance transform')
axes[1, 0].axis('off')

# Seeds
axes[1, 1].imshow(binary_clean, cmap='gray')
axes[1, 1].scatter(coords[:, 1], coords[:, 0], c='red', s=20, marker='+')
axes[1, 1].set_title(f'Seeds ({num_seeds} found)')
axes[1, 1].axis('off')

# Final result with boundaries and centroids
axes[1, 2].imshow(img_channel, cmap='gray')
# Draw cell boundaries in cyan
boundaries = segmentation.find_boundaries(labels, mode='thick')
axes[1, 2].imshow(np.ma.masked_where(~boundaries, boundaries), cmap='cyan', alpha=0.6)
# Mark centroids in red
for region in valid_cells:
    y, x = region.centroid
    axes[1, 2].plot(x, y, 'r+', markersize=10, markeredgewidth=2)
axes[1, 2].set_title(f'Detected cells: {cell_count}')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Optional: Save the labeled image for downstream analysis
# labeled_image = labels  # Each cell has a unique integer ID

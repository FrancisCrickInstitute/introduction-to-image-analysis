import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature

# ---------------- Settings ----------------
NUCLEI_CHANNEL = 0    # index of the channel to count (e.g. DAPI/Hoechst)
SMOOTH_SIGMA   = 2    # Gaussian smoothing (px) to suppress noise
MIN_AREA       = 30   # px; discard objects smaller than this (debris/noise)
MIN_DISTANCE   = 5    # px; minimum spacing between cell centres for splitting
# ------------------------------------------

# 1. Make the channel axis the last axis
arr = np.asarray(img)
if arr.ndim == 2:
    arr = arr[..., np.newaxis]
elif arr.ndim == 3:
    # The channel axis is assumed to be the smallest dimension
    arr = np.moveaxis(arr, int(np.argmin(arr.shape)), -1)
else:
    raise ValueError(f"Expected a 2D multichannel image, got shape {arr.shape}")

print("Image shape (H, W, C):", arr.shape)

# 2. Extract and normalise the nuclei channel
nuc = arr[..., NUCLEI_CHANNEL].astype(float)
nuc = (nuc - nuc.min()) / (nuc.max() - nuc.min() + 1e-12)

# 3. Smooth and threshold
smooth = filters.gaussian(nuc, sigma=SMOOTH_SIGMA)
thresh = filters.threshold_otsu(smooth)
mask = smooth > thresh
mask = ndi.binary_fill_holes(mask)
mask = morphology.opening(mask, morphology.disk(2))
mask = morphology.remove_small_objects(mask, min_size=MIN_AREA)

# 4. Split touching cells with a distance-transform watershed
distance = ndi.distance_transform_edt(mask)
coords = feature.peak_local_max(
    distance,
    min_distance=MIN_DISTANCE,
    labels=measure.label(mask),
    exclude_border=False,
)
markers = np.zeros(mask.shape, dtype=np.int32)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
labels = segmentation.watershed(-distance, markers, mask=mask)

# 5. Remove small fragments and relabel
labels = morphology.remove_small_objects(labels, min_size=MIN_AREA)
labels, _, _ = segmentation.relabel_sequential(labels)

cell_count = labels.max()
print(f"Cell count: {cell_count}")

# 6. Visualise the result
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(nuc, cmap="gray")
axes[0].set_title(f"Channel {NUCLEI_CHANNEL}")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Foreground mask")
axes[2].imshow(segmentation.mark_boundaries(nuc, labels, color=(1, 0, 0)))
axes[2].set_title(f"Segmented: {cell_count} cells")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

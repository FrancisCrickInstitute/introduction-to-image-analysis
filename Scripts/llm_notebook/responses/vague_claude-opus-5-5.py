import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, segmentation, measure, feature, exposure

# ----------------------------- Parameters ---------------------------------
NUCLEAR_CHANNEL   = 0      # index of the nuclear stain channel (DAPI/Hoechst)
CHANNEL_AXIS      = None   # None = auto-detect (smallest axis); or set 0 / -1 explicitly
SMOOTH_SIGMA      = 2      # Gaussian blur (pixels) before thresholding
MIN_NUCLEUS_AREA  = 50     # objects smaller than this (pixels) are discarded as debris
MIN_DISTANCE      = 7      # min distance (pixels) between nucleus centres; ~ nucleus radius
EXCLUDE_BORDER    = False  # True = don't count nuclei touching the image edge
# --------------------------------------------------------------------------


def get_channel(image, channel, channel_axis=None):
    """Extract a single 2D channel from a 2D or multichannel image."""
    image = np.squeeze(np.asarray(image))
    if image.ndim == 2:
        return image.astype(float)
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D multichannel image, got shape {image.shape}")
    if channel_axis is None:
        channel_axis = int(np.argmin(image.shape))  # channels = smallest axis
    print(f"Image shape {image.shape}; using axis {channel_axis} as channels "
          f"({image.shape[channel_axis]} channels).")
    return np.take(image, channel, axis=channel_axis).astype(float)


def remove_small_labels(label_img, min_area):
    """Remove labelled objects with area < min_area."""
    sizes = np.bincount(label_img.ravel())
    too_small = sizes < min_area
    too_small[0] = False  # keep background as-is
    label_img = label_img.copy()
    label_img[too_small[label_img]] = 0
    return label_img


# 1. Extract and normalise the nuclear channel (robust to outlier pixels)
nuc = get_channel(img, NUCLEAR_CHANNEL, CHANNEL_AXIS)
lo, hi = np.percentile(nuc, (0.5, 99.8))
nuc = exposure.rescale_intensity(nuc, in_range=(lo, hi), out_range=(0, 1))

# 2. Smooth and threshold (Otsu) to get a foreground mask
smooth = filters.gaussian(nuc, sigma=SMOOTH_SIGMA)
mask = smooth > filters.threshold_otsu(smooth)
mask = ndi.binary_fill_holes(mask)
mask = remove_small_labels(measure.label(mask), MIN_NUCLEUS_AREA) > 0

# 3. Split touching nuclei: watershed on the distance transform
distance = ndi.distance_transform_edt(mask)
distance = filters.gaussian(distance, sigma=1)  # suppress spurious peaks
peaks = feature.peak_local_max(
    distance,
    min_distance=MIN_DISTANCE,
    labels=measure.label(mask),
    exclude_border=False,
)
markers = np.zeros(mask.shape, dtype=np.int32)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
labels = segmentation.watershed(-distance, markers, mask=mask)

# 4. Clean up: drop small fragments, optionally edge-touching nuclei
labels = remove_small_labels(labels, MIN_NUCLEUS_AREA)
if EXCLUDE_BORDER:
    labels = segmentation.clear_border(labels)
labels, _, _ = segmentation.relabel_sequential(labels)

n_cells = int(labels.max())
print(f"Number of cells (nuclei) detected: {n_cells}")

# 5. Per-cell measurements (handy for QC / downstream analysis)
props = measure.regionprops_table(
    labels, intensity_image=nuc,
    properties=("label", "area", "centroid", "eccentricity", "mean_intensity"),
)
try:
    import pandas as pd
    cell_table = pd.DataFrame(props)
    display(cell_table.describe())
except ImportError:
    cell_table = props

# 6. Visual check
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(nuc, cmap="gray")
axes[0].set_title(f"Nuclear channel ({NUCLEAR_CHANNEL})")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Foreground mask")
axes[2].imshow(segmentation.mark_boundaries(nuc, labels, color=(1, 0, 0)))
centroids = np.array([r.centroid for r in measure.regionprops(labels)])
if len(centroids):
    axes[2].plot(centroids[:, 1], centroids[:, 0], "y+", markersize=6)
axes[2].set_title(f"Segmentation: {n_cells} cells")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

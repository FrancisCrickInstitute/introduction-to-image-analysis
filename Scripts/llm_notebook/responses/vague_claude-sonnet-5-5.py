import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature, exposure

# ----------------- Parameters to adjust -----------------
NUCLEAR_CHANNEL = 0       # index of the channel with one blob per cell (e.g. DAPI)
SMOOTH_SIGMA = 2          # Gaussian blur (px); ~ 1/5 of a cell diameter
BG_SIGMA = 50             # large-scale background estimate (px); set None to skip
MIN_DISTANCE = 7          # min distance between cell centres (px); ~ cell radius
MIN_AREA = 30             # discard objects smaller than this (px²)
MAX_AREA = None           # discard objects larger than this (px²), None = no limit
REMOVE_BORDER_CELLS = False   # True to exclude cells touching the image edge
# ---------------------------------------------------------

# --- 1. Select the channel (handles channels-first or channels-last) ---
arr = np.asarray(img)
if arr.ndim == 3:
    ch_axis = int(np.argmin(arr.shape))          # channel axis = smallest dimension
    print(f"Image shape {arr.shape}, assuming channel axis = {ch_axis}")
    nuc = np.take(arr, NUCLEAR_CHANNEL, axis=ch_axis)
elif arr.ndim == 2:
    nuc = arr
else:
    raise ValueError(f"Expected a 2D (multichannel) image, got shape {arr.shape}")

nuc = nuc.astype(float)
nuc = (nuc - nuc.min()) / (np.ptp(nuc) + 1e-12)   # scale to 0–1

# --- 2. Denoise and remove uneven background ---
smooth = filters.gaussian(nuc, sigma=SMOOTH_SIGMA)
if BG_SIGMA:
    smooth = smooth - filters.gaussian(nuc, sigma=BG_SIGMA)
    smooth = np.clip(smooth, 0, None)

# --- 3. Threshold to get foreground mask ---
thresh = filters.threshold_otsu(smooth)
mask = smooth > thresh
mask = morphology.remove_small_objects(mask, MIN_AREA)
mask = ndi.binary_fill_holes(mask)

# --- 4. Split touching cells with a distance-transform watershed ---
distance = ndi.distance_transform_edt(mask)
distance = filters.gaussian(distance, sigma=1)
peaks = feature.peak_local_max(
    distance, min_distance=MIN_DISTANCE, labels=measure.label(mask), exclude_border=False
)
markers = np.zeros_like(mask, dtype=int)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
labels = segmentation.watershed(-distance, markers, mask=mask)

# --- 5. Filter objects by size / border ---
if REMOVE_BORDER_CELLS:
    labels = segmentation.clear_border(labels)

for region in measure.regionprops(labels):
    too_small = region.area < MIN_AREA
    too_big = MAX_AREA is not None and region.area > MAX_AREA
    if too_small or too_big:
        labels[labels == region.label] = 0

labels, _, _ = segmentation.relabel_sequential(labels)
n_cells = labels.max()
areas = [r.area for r in measure.regionprops(labels)]

print(f"Cell count: {n_cells}")
print(f"Area (px²): median={np.median(areas):.0f}, "
      f"min={np.min(areas):.0f}, max={np.max(areas):.0f}")

# --- 6. Visual check ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
axes[0].imshow(exposure.rescale_intensity(nuc, out_range=(0, 1)), cmap="gray")
axes[0].set_title("Selected channel")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Foreground mask")
axes[2].imshow(nuc, cmap="gray")
axes[2].contour(labels > 0, levels=[0.5], colors="yellow", linewidths=0.5)
axes[2].imshow(np.ma.masked_equal(segmentation.find_boundaries(labels), 0),
               cmap="autumn", alpha=1)
axes[2].set_title(f"Segmentation: {n_cells} cells")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

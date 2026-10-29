import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max

# ---------- parameters to tune ----------
nucleus_channel = 0        # channel containing nuclei/cells, e.g. DAPI
sigma = 1.0                # Gaussian smoothing
min_cell_area = 50         # minimum object size in pixels
max_cell_area = None       # e.g. 5000; None = no upper limit
min_peak_distance = 5      # roughly half the minimum nucleus radius
invert_intensity = False   # True if cells are dark on a bright background
# ----------------------------------------

def get_2d_channel(img, channel=0):
    """Return a single 2D channel from 2D or multichannel image."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(float)
    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {a.shape}")

    # Common cases: H,W,C or C,H,W
    if a.shape[-1] <= 5:       # H, W, C
        return a[..., channel].astype(float)
    elif a.shape[0] <= 5:      # C, H, W
        return a[channel].astype(float)
    else:
        raise ValueError("Could not determine channel axis; edit get_2d_channel manually.")

ch = get_2d_channel(img, nucleus_channel)

if invert_intensity:
    ch = ch.max() - ch

# Smooth and threshold
ch_smooth = filters.gaussian(ch, sigma=sigma, preserve_range=True)
thr = filters.threshold_otsu(ch_smooth)
mask = ch_smooth > thr

# Remove noise/small holes
mask = morphology.remove_small_objects(mask, min_size=min_cell_area)
mask = morphology.remove_small_holes(mask, area_threshold=min_cell_area)
mask = morphology.binary_opening(mask, morphology.disk(1))

# Distance transform + watershed to separate touching nuclei
distance = ndi.distance_transform_edt(mask)
coords = peak_local_max(
    distance,
    min_distance=min_peak_distance,
    labels=mask,
    exclude_border=False
)

markers = np.zeros(distance.shape, dtype=np.int32)
if len(coords) > 0:
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

labels = segmentation.watershed(-distance, markers, mask=mask)

# Filter objects by area
props = measure.regionprops(labels)
keep = []
for p in props:
    if p.area >= min_cell_area and (max_cell_area is None or p.area <= max_cell_area):
        keep.append(p.label)

# Relabel only kept objects
labels_clean = np.zeros_like(labels, dtype=np.int32)
for new_id, old_id in enumerate(keep, start=1):
    labels_clean[labels == old_id] = new_id

cell_count = len(keep)
print(f"Estimated cell count: {cell_count}")

# Visualization
disp = exposure.rescale_intensity(ch, out_range=(0, 1))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(disp, cmap="gray")
ax[0].set_title(f"Channel {nucleus_channel}")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Foreground mask")

ax[2].imshow(disp, cmap="gray")
ax[2].imshow(
    np.ma.masked_where(labels_clean == 0, labels_clean),
    cmap="nipy_spectral",
    alpha=0.35
)
ax[2].contour(labels_clean, colors="red", linewidths=0.4)
ax[2].set_title(f"Segmentation: {cell_count} cells")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation
from skimage.feature import peak_local_max

# --- Parameters you may need to tune ---
channel = None          # Set to e.g. 0 to use a specific channel; None uses max projection
sigma = 1.5             # Gaussian smoothing sigma
min_size = 100          # Minimum cell area in pixels
min_distance = 10       # Minimum distance between cell centres in pixels

img = np.asarray(img)

# --- Reduce multichannel image to a single 2D image ---
if img.ndim == 3:
    # Detect channels-first (C,H,W) vs channels-last (H,W,C)
    if img.shape[0] <= 10 and img.shape[-1] > 10:
        # channels-first
        if channel is None:
            image_2d = np.max(img, axis=0)
        else:
            image_2d = img[channel]
    elif img.shape[-1] <= 10 and img.shape[0] > 10:
        # channels-last
        if channel is None:
            image_2d = np.max(img, axis=2)
        else:
            image_2d = img[..., channel]
    else:
        # Ambiguous: assume channels-last if last axis is the smallest
        if img.shape[-1] <= img.shape[0]:
            if channel is None:
                image_2d = np.max(img, axis=2)
            else:
                image_2d = img[..., channel]
        else:
            if channel is None:
                image_2d = np.max(img, axis=0)
            else:
                image_2d = img[channel]
elif img.ndim == 2:
    image_2d = img
else:
    raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

# --- Normalise to [0, 1] ---
image_2d = image_2d.astype(np.float32)
image_2d = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)

# --- Smooth and threshold ---
image_smooth = filters.gaussian(image_2d, sigma=sigma)
thresh = filters.threshold_otsu(image_smooth)
mask = image_smooth > thresh       # Use < if cells are dark on bright background

# --- Clean mask ---
mask = morphology.remove_small_objects(mask, min_size=min_size)
mask = ndi.binary_fill_holes(mask)
mask = morphology.remove_small_objects(mask, min_size=min_size)

# --- Watershed to split touching cells ---
if mask.sum() > 0:
    distance = ndi.distance_transform_edt(mask)
    distance_smooth = filters.gaussian(distance, sigma=sigma * 1.5)

    coords = peak_local_max(distance_smooth, min_distance=min_distance, exclude_border=False)
    markers = np.zeros(distance_smooth.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = segmentation.watershed(-distance_smooth, markers, mask=mask)
    n_cells = labels.max()
else:
    labels = np.zeros(image_2d.shape, dtype=np.int32)
    n_cells = 0

print(f"Number of cells: {n_cells}")
# labels now contains the labelled cell segmentation image

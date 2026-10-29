import numpy as np
from scipy import ndimage as ndi
from skimage import filters, feature, morphology, segmentation

# 1. Denoise and threshold (Otsu works well for bimodal fluorescence)
smooth = filters.gaussian(img, sigma=1.5)
mask   = smooth > filters.threshold_otsu(smooth)

# 2. Clean up: fill holes and drop tiny debris
mask = ndi.binary_fill_holes(mask)
mask = morphology.remove_small_objects(mask, min_size=50)

# 3. Separate touching nuclei with a watershed on the distance map
dist   = ndi.distance_transform_edt(mask)
coords = feature.peak_local_max(dist, min_distance=10, labels=mask)
markers = np.zeros(dist.shape, dtype=int)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
labels = segmentation.watershed(-dist, markers, mask=mask)

# 4. Count (label 0 is background)
n_nuclei = labels.max()
print(f"Nuclei counted: {n_nuclei}")

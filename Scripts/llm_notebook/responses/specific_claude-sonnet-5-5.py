import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, feature

# Smooth and subtract uneven background
sm = filters.gaussian(img.astype(float), sigma=1)
bg = filters.gaussian(sm, sigma=50)
corr = sm - bg

# Otsu threshold, then clean up the mask
mask = corr > filters.threshold_otsu(corr)
mask = morphology.remove_small_objects(mask, 30)
mask = ndi.binary_fill_holes(mask)

# Watershed split of touching nuclei using the distance transform
dist = ndi.distance_transform_edt(mask)
dist_s = filters.gaussian(dist, sigma=1)
peaks = feature.peak_local_max(dist_s, min_distance=5, labels=measure.label(mask),
                               exclude_border=False)
markers = np.zeros(img.shape, dtype=int)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
labels = segmentation.watershed(-dist_s, markers, mask=mask)

print("Nuclei count:", labels.max())

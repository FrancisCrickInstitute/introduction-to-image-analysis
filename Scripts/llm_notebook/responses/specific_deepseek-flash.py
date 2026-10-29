import numpy as np
from skimage import filters, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Smooth and threshold
sm = filters.gaussian(img, sigma=1, preserve_range=True)
mask = sm > filters.threshold_otsu(sm)

# Clean mask
mask = morphology.remove_small_objects(mask, min_size=30)
mask = morphology.binary_closing(mask, morphology.disk(2))
mask = ndi.binary_fill_holes(mask)

# Split touching nuclei with watershed
dist = ndi.distance_transform_edt(mask)
peaks = peak_local_max(dist, min_distance=10, labels=mask, exclude_border=False)

markers = np.zeros(mask.shape, dtype=np.int32)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)

labels = segmentation.watershed(-dist, markers, mask=mask)
nuclei_count = labels.max()

print(nuclei_count)

import numpy as np
from skimage import filters, morphology, measure, segmentation, feature

img_f = img.astype(float)

# 1. Smooth and threshold (Otsu) to get the foreground mask
sm = filters.gaussian(img_f, sigma=2, preserve_range=True)
mask = sm > filters.threshold_otsu(sm)
mask = morphology.remove_small_holes(mask, area_threshold=30)
mask = morphology.remove_small_objects(mask, min_size=30)

# 2. One marker per nucleus: local maxima of a more heavily smoothed image
sm_marker = filters.gaussian(img_f, sigma=3, preserve_range=True)
coords = feature.peak_local_max(sm_marker, min_distance=7,
                                labels=measure.label(mask))
markers = np.zeros(img.shape, dtype=np.int32)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

# 3. Watershed to split touching nuclei
labels = segmentation.watershed(-sm, markers, mask=mask)

# 4. Drop tiny fragments and count
n_nuclei = sum(r.area >= 30 for r in measure.regionprops(labels))
print(f"Nuclei counted: {n_nuclei}")

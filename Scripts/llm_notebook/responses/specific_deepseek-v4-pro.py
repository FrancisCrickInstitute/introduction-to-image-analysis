import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature

# Smooth and threshold
img_smooth = filters.gaussian(img, sigma=1.5)
threshold = filters.threshold_otsu(img_smooth)
mask = img_smooth > threshold

# Clean up
mask = morphology.remove_small_objects(mask, min_size=64)
mask = morphology.remove_small_holes(mask, area_threshold=64)

# Separate touching nuclei using distance transform + watershed
distance = ndi.distance_transform_edt(mask)
coords = feature.peak_local_max(distance, min_distance=10, labels=mask)

markers = np.zeros(img.shape, dtype=np.int32)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

labels = segmentation.watershed(-distance, markers, mask=mask)

nuclei_count = labels.max()
print(f"Number of nuclei: {nuclei_count}")

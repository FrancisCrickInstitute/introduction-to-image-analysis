import numpy as np
from skimage import color, filters, morphology, measure
from skimage.util import img_as_float

# Convert to grayscale float in [0, 1]
gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img_as_float(img)

# Otsu threshold; invert if background was selected as foreground
mask = gray > filters.threshold_otsu(gray)
if mask.mean() > 0.5:
    mask = ~mask

# Clean up mask and count connected components
mask = morphology.remove_small_objects(mask, min_size=64, connectivity=2)
mask = morphology.remove_small_holes(mask, area_threshold=64, connectivity=2)
labels = measure.label(mask, connectivity=2)

count = labels.max()
print(count)

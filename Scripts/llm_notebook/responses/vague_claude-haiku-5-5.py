import numpy as np
from skimage import color, filters, measure, morphology

# Convert to grayscale float (drop alpha channel if present)
gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float)

# Smooth, threshold (Otsu), and make sure objects are the minority class
gray = filters.gaussian(gray, sigma=1)
mask = gray > filters.threshold_otsu(gray)
if mask.mean() > 0.5:
    mask = ~mask

# Clean up noise and label connected components
mask = morphology.remove_small_objects(mask, min_size=20)
mask = morphology.remove_small_holes(mask, area_threshold=20)
labels = measure.label(mask)

print(f"Number of objects: {labels.max()}")

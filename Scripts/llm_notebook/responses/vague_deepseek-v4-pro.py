import numpy as np
from skimage import color, filters, measure, morphology

# Convert to grayscale if needed
gray = color.rgb2gray(img) if img.ndim == 3 else img

# Otsu threshold
thresh = filters.threshold_otsu(gray)
mask = gray > thresh

# Assume objects are the smaller area class (background covers more area)
if mask.mean() > 0.5:
    mask = ~mask

# Remove small specks (adjust min_size as needed)
mask = morphology.remove_small_objects(mask, min_size=32)

# Label connected components and count
labels = measure.label(mask)
n_objects = np.max(labels)  # background = 0

print(f"Number of objects: {n_objects}")

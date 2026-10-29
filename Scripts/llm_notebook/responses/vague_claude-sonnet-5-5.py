import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, measure, segmentation

# Grayscale + smooth
gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float)
gray = filters.gaussian(gray, sigma=1)

# Threshold (Otsu); ensure objects are the minority class
mask = gray > filters.threshold_otsu(gray)
if mask.mean() > 0.5:
    mask = ~mask

# Clean up
min_size = 30  # adjust to your object size (pixels)
mask = morphology.remove_small_objects(mask, min_size)
mask = morphology.remove_small_holes(mask, min_size)

# Label and count
labels = measure.label(mask)
print("Object count:", labels.max())

# Visual check
plt.imshow(segmentation.mark_boundaries(img if img.ndim == 3 else gray, labels))
plt.title(f"{labels.max()} objects"); plt.axis("off"); plt.show()

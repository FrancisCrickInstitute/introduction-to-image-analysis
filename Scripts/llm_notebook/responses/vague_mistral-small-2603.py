import numpy as np
from skimage.measure import label, regionprops

# Assuming img is already loaded (grayscale or binary)
binary = img > threshold_otsu(img) if img.dtype != bool else img  # Auto-threshold if needed
labeled = label(binary)
objects = regionprops(labeled)

print(f"Number of objects: {len(objects)}")

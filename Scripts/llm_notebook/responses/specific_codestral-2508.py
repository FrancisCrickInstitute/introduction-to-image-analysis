from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label
import numpy as np

# Preprocess the image
blurred = gaussian(img, sigma=1)
thresh = threshold_otsu(blurred)
binary = blurred > thresh

# Clean up the binary image
cleaned = remove_small_objects(binary, min_size=20)
cleaned = binary_closing(cleaned, footprint=disk(2))

# Count nuclei
labeled = label(cleaned)
nuclei_count = labeled.max()

print(f"Number of nuclei: {nuclei_count}")

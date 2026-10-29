import numpy as np
from skimage import filters, morphology, measure

# Threshold using Otsu's method
threshold = filters.threshold_otsu(img)
binary = img > threshold

# Separate touching nuclei with watershed
distance = morphology.distance_transform_edt(binary)
coords = measure.label(binary)
peaks = morphology.local_maxima(distance, indices=True)
markers = np.zeros_like(img, dtype=np.uint8)
markers[peaks] = 1
markers = measure.label(markers)
watershed = morphology.watershed(-distance, markers, mask=binary)

# Count nuclei
nuclei_count = measure.label(watershed).max()

print(f"Number of nuclei: {nuclei_count}")

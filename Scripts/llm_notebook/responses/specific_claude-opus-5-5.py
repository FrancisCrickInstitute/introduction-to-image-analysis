import numpy as np
from scipy import ndimage as ndi  # installed with scikit-image
from skimage import filters, morphology, measure, segmentation, feature, color
import matplotlib.pyplot as plt

# 1. Smooth and threshold (Otsu)
sm = filters.gaussian(img.astype(float), sigma=2)
mask = sm > filters.threshold_otsu(sm)
mask = ndi.binary_fill_holes(morphology.remove_small_objects(mask, 50))

# 2. Split touching nuclei: distance transform + watershed
dist = filters.gaussian(ndi.distance_transform_edt(mask), sigma=1)
peaks = feature.peak_local_max(dist, min_distance=7, labels=measure.label(mask),
                               exclude_border=False)
markers = np.zeros(mask.shape, int)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
labels = segmentation.watershed(-dist, markers, mask=mask)

# 3. Drop tiny fragments and count
labels = morphology.remove_small_objects(labels, 50)
n = len(np.unique(labels)) - 1
print(f"Nuclei count: {n}")

# 4. Visual check
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(img, cmap='gray'); ax[0].set_title('Image')
ax[1].imshow(color.label2rgb(labels, image=sm, bg_label=0)); ax[1].set_title(f'{n} nuclei')
for a in ax: a.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from skimage.segmentation import mark_boundaries

# If filename is not already defined, set a fallback.
try:
    filename
except NameError:
    filename = "image.tif"

# --- Identify channel layout ---
# Assume: channel 0 = nuclei, 1 = tubulin, 2 = actin, 3 = YAP/TAZ.
# Handle both (C, H, W) and (H, W, C) array layouts.
if img.ndim == 3 and img.shape[0] <= 4:
    nuclear = img[0]
    tubulin = img[1]
    actin = img[2]
else:
    nuclear = img[..., 0]
    tubulin = img[..., 1]
    actin = img[..., 2]

# --- Simple global grey-level thresholding using Otsu ---
thresh = filters.threshold_otsu(nuclear)
mask = nuclear > thresh

# --- Clean up binary mask ---
mask = morphology.remove_small_objects(mask, min_size=64)
mask = morphology.remove_small_holes(mask, area_threshold=64)

# --- Count cells/nuclei as connected components ---
labels = measure.label(mask, connectivity=2)
n_cells = labels.max()

# --- Build RGB composite: nuclei=blue, tubulin=green, actin=red ---
def normalize(ch):
    ch = ch.astype(np.float32)
    ch -= ch.min()
    ch /= ch.max() + 1e-8
    return ch

composite = np.stack([normalize(actin),
                      normalize(tubulin),
                      normalize(nuclear)], axis=-1)

# --- Overlay segmentation boundaries on the composite ---
overlay = mark_boundaries(composite, labels, color=(1, 1, 0), mode='thick')

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title(f"Segmented cells: {n_cells}")
plt.axis("off")
plt.show()

# --- Markdown table ---
print("| Input filename | Thresholding method | Thresholding value | Number of cells | Notes |")
print("| --- | --- | --- | --- | --- |")
print(f"| {filename} | Otsu (global grey-level) | {thresh:.2f} | {n_cells} | Nuclear channel (ch 0); touching nuclei are not split; YAP/TAZ not used |")

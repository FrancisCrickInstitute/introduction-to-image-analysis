import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation, color

# --- 1. Select the nuclear channel (Hoechst = channel 1) ---
# Channel-first (4, H, W) or channel-last (H, W, 4)? Adjust if needed.
nuc = img[0] if img.shape[0] == 4 else img[..., 0]
nuc = nuc.astype(float)

# --- 2. Smooth to reduce noise, then apply a global Otsu threshold ---
smooth = filters.gaussian(nuc, sigma=2, preserve_range=True)  # sigma ~ nucleus size / 10
thr = filters.threshold_otsu(smooth)                          # automatic grey-level threshold
mask = smooth > thr                                           # binary: nucleus vs background

# --- 3. Clean the mask ---
min_size = 50                                                 # px; removes debris, adjust to nucleus size
mask = morphology.remove_small_objects(mask, min_size=min_size)
mask = morphology.remove_small_holes(mask)                    # fill holes inside nuclei

# --- 4. Label connected blobs: each blob = one nucleus ---
labels = measure.label(mask)
n_cells = labels.max()                                        # number of labelled objects

# --- 5. Overlay: red outlines of segmented nuclei on the nuclear image ---
disp = (nuc - nuc.min()) / (nuc.max() - nuc.min())            # rescale to 0-1 for display
overlay = color.gray2rgb(disp)
overlay[segmentation.find_boundaries(labels)] = [1, 0, 0]     # paint boundaries red

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(overlay)
ax.set_title(f'{filename}: {n_cells} cells')
ax.axis('off')
plt.savefig(filename.rsplit('.', 1)[0] + '_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 6. Results table in Markdown ---
areas = [r.area for r in measure.regionprops(labels)]        # area of each nucleus (px)
median_area = np.median(areas) if areas else 0
notes = (f"Gaussian sigma=2; min object size={min_size} px; "
         f"median nucleus area={median_area:.0f} px; "
         "touching nuclei not split (no watershed)")

table = f"""| Input filename | Thresholding method | Thresholding value | Number of cells | Notes |
|---|---|---|---|---|
| {filename} | Otsu (global) | {thr:.1f} | {n_cells} | {notes} |"""
print(table)

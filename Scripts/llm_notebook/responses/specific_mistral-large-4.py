# ============================================================
#  Cell counting from a 4‑channel fluorescence TIF
#  Channel order: 0=Nuclei(Hoechst), 1=Tubulin, 2=Actin, 3=YAP/TAZ
#  Pre‑existing variables: img (numpy array), filename (str)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, segmentation, color

# --- 1. Grab the nuclei channel ---------------------------------
# If img.shape is (H, W, 4) instead of (4, H, W), uncomment:
# img = np.moveaxis(img, -1, 0)
nuclei = img[0].astype(float)          # channel 0 = Hoechst

# --- 2. Simple grey‑level thresholding (Otsu) -------------------
thresh_val = filters.threshold_otsu(nuclei)   # automatic global threshold
binary = nuclei > thresh_val                   # boolean mask of nuclei

# --- 3. Count cells as connected components ----------------------
labels = measure.label(binary)         # each nucleus gets a unique integer ID
num_cells = labels.max()               # label 0 is background → max ID = cell count

# --- 4. Overlay segmentation on the original image --------------
# Build a quick grayscale background from the max‑projection of all 4 channels
bg = img.max(axis=0)
bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)   # normalise to 0‑1

rgb = color.gray2rgb(bg)               # convert to RGB so we can colour the outlines
edges = segmentation.find_boundaries(labels, mode='outer')
rgb[edges] = [1, 0, 0]                 # draw cell borders in red

# Display original and overlay side‑by‑side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(bg, cmap='gray'); ax[0].set_title('Original (max projection)'); ax[0].axis('off')
ax[1].imshow(rgb);         ax[1].set_title(f'Segmented cells: {num_cells}'); ax[1].axis('off')
plt.tight_layout()
plt.show()

# --- 5. Markdown results table -----------------------------------
print("| Input filename | Thresholding method | Thresholding value | Number of cells | Notes |")
print("|---|---|---|---|---|")
print(f"| {filename} | Otsu | {thresh_val:.2f} | {num_cells} "
      f"| Hoechst channel; connected components; no size filter |")

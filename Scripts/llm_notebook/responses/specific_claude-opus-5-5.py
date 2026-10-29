# ============================================================
# Nuclear segmentation & cell counting by grey-level thresholding
# Assumes `img` (4-channel image) and `filename` already exist
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, color, exposure
from IPython.display import Markdown, display

# --- Parameters (adjust to suit your image resolution) ---
SIGMA = 2          # Gaussian blur radius (pixels) to suppress noise before thresholding
MIN_AREA = 50      # Objects smaller than this (pixels) are treated as debris and removed

# --- 1. Extract the nuclear channel (channel 1 = Hoechst) ---
# The channel axis is assumed to be the smallest dimension (size 4).
# This works for both (C, Y, X) and (Y, X, C) layouts.
ch_axis = int(np.argmin(img.shape))
nuc = np.take(img, 0, axis=ch_axis).astype(float)

# --- 2. Smooth the image to reduce noise ---
nuc_smooth = filters.gaussian(nuc, sigma=SIGMA, preserve_range=True)

# --- 3. Global grey-level threshold using Otsu's method ---
# Otsu picks the value that best separates foreground (nuclei)
# from background, based on the intensity histogram.
thresh = filters.threshold_otsu(nuc_smooth)
mask = nuc_smooth > thresh

# --- 4. Clean up the binary mask ---
mask = ndi.binary_fill_holes(mask)                     # fill holes inside nuclei
mask = morphology.remove_small_objects(mask, MIN_AREA) # remove small debris

# --- 5. Label connected objects (each object = one nucleus/cell) ---
labels = measure.label(mask)
n_cells = labels.max()

# Count how many nuclei touch the image edge (these may be partial cells)
n_border = n_cells - len(np.unique(segmentation.clear_border(labels))) + 1

# --- 6. Create the overlay ---
# Contrast-stretch the nuclear channel for display (1st-99th percentile)
p1, p99 = np.percentile(nuc, (1, 99))
nuc_disp = exposure.rescale_intensity(nuc, in_range=(p1, p99), out_range=(0, 1))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(nuc_disp, cmap='gray')
ax[0].set_title('Nuclear channel (Hoechst)')

# Coloured labels blended onto the image, with outlines drawn in yellow
overlay = color.label2rgb(labels, image=nuc_disp, alpha=0.3, bg_label=0)
overlay = segmentation.mark_boundaries(overlay, labels, color=(1, 1, 0))
ax[1].imshow(overlay)
ax[1].set_title(f'Segmented nuclei (n = {n_cells})')

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig('segmentation_overlay.png', dpi=150)  # also save overlay to disk
plt.show()

# --- 7. Summary table in markdown format ---
notes = (f"Image shape {img.shape}; Gaussian sigma={SIGMA}; min area={MIN_AREA} px; "
         f"{n_border} nuclei touch the border; touching nuclei are not split, "
         f"so the count may be an underestimate")

table = (
    "| Filename | Threshold method | Threshold value | Number of cells | Notes |\n"
    "|---|---|---|---|---|\n"
    f"| {filename} | Otsu (global) | {thresh:.2f} | {n_cells} | {notes} |"
)
display(Markdown(table))  # rendered table in the notebook
print(table)              # raw markdown text for copy/paste

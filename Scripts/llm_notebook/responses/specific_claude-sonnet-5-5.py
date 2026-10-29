import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation, exposure
from IPython.display import display, Markdown

# --- 1. Select the nuclear channel (channel 1, Hoechst) ---
# Make sure channels come first: (4, H, W). If they come last (H, W, 4), move them.
im = np.moveaxis(img, -1, 0) if img.shape[-1] == 4 else img
nuc = im[0]

# --- 2. Pre-process: light Gaussian blur to suppress noise ---
# preserve_range keeps the original grey-level units, so the threshold value is interpretable
smooth = filters.gaussian(nuc, sigma=1, preserve_range=True)

# --- 3. Simple global grey-level threshold (Otsu's method) ---
thresh = filters.threshold_otsu(smooth)
mask = smooth > thresh

# --- 4. Clean up the binary mask ---
min_size = 30                                        # smallest object kept (pixels)
mask = morphology.remove_small_objects(mask, min_size=min_size)  # remove debris/noise
mask = morphology.remove_small_holes(mask, min_size)             # fill small holes in nuclei

# --- 5. Label connected components: each one is counted as one cell ---
labels = measure.label(mask)
n_cells = labels.max()

# --- 6. Overlay the segmentation outlines on the original (nuclear) image ---
disp = exposure.rescale_intensity(nuc.astype(float), out_range=(0, 1))  # contrast stretch for display
overlay = segmentation.mark_boundaries(disp, labels, color=(1, 0, 0))   # red outlines
plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title(f"Nuclear channel with segmented cells (n = {n_cells})")
plt.axis("off")
plt.show()

# --- 7. Results table in markdown format ---
notes = (f"Gaussian blur (sigma=1) applied before thresholding; objects < {min_size} px removed; "
         "no splitting of touching nuclei, so clusters count as one cell and the count may be underestimated; "
         "edge-touching nuclei are included")
table = (
    "| Filename | Thresholding method | Thresholding value | Number of cells | Notes |\n"
    "|---|---|---|---|---|\n"
    f"| {filename} | Otsu (global) | {thresh:.2f} | {n_cells} | {notes} |"
)
display(Markdown(table))

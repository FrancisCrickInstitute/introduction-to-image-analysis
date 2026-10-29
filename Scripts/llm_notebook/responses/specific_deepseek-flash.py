# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from skimage import filters, measure, color

# --- Ensure the 4-channel image is in (Y, X, C) order ---
if img.ndim == 3 and img.shape[-1] == 4:       # already YXC
    ch = img
elif img.ndim == 3 and img.shape[0] == 4:      # CYX -> YXC
    ch = np.moveaxis(img, 0, -1)
else:
    raise ValueError("Expected a 3D image with 4 channels (YXC or CYX).")

# --- Nuclear channel: Hoechst is channel 0 ---
nuc = ch[..., 0]

# --- Simple grey-level thresholding using Otsu's method ---
thr = filters.threshold_otsu(nuc)              # automatic threshold value
mask = nuc > thr                               # binary nuclear mask

# --- Count connected nuclear regions as cells ---
labels = measure.label(mask)                   # label connected components
n_cells = labels.max()                         # number of labelled objects

# --- Make an RGB composite for the overlay background ---
def norm(x):                                   # scale each channel to 0-1 for display
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# RGB = actin (red), tubulin (green), nuclei (blue); YAP/TAZ not shown
rgb = np.dstack([norm(ch[..., 2]), norm(ch[..., 1]), norm(ch[..., 0])])

# --- Overlay segmented nuclei on the RGB composite ---
overlay = color.label2rgb(labels, image=rgb, bg_label=0, alpha=0.35)

# --- Display overlay ---
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title(f"Segmented nuclei: Otsu={thr:.3g}, cells={n_cells}")
plt.axis("off")
plt.show()

# --- Markdown results table ---
notes = ("Hoechst channel thresholded by Otsu; connected-component count; "
         "touching nuclei are not split; overlay uses actin/tubulin/Hoechst RGB.")
display(Markdown(f"""
| Input filename | Thresholding method | Thresholding value | Number of cells | Notes |
|---|---:|---:|---:|---|
| `{filename}` | Otsu (global grey-level) | {thr:.3g} | {n_cells} | {notes} |
"""))

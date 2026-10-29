import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, measure

# Convert to grayscale (drop alpha channel if present) and smooth
g = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float)
g = filters.gaussian(g, sigma=1)

# Otsu threshold; assume objects cover less area than background
mask = g > filters.threshold_otsu(g)
if mask.mean() > 0.5:
    mask = ~mask

# Clean up: remove specks and fill small holes (tune 50 to your object size)
mask = morphology.remove_small_objects(mask, 50)
mask = morphology.remove_small_holes(mask, 50)

# Label connected components and count
labels = measure.label(mask)
n = labels.max()
print(f"Object count: {n}")

# Quick visual check
plt.imshow(color.label2rgb(labels, image=g, bg_label=0))
plt.title(f"{n} objects")
plt.axis("off")
plt.show()

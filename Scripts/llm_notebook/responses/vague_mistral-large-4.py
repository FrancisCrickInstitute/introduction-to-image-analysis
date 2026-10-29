import numpy as np
from skimage import color, filters, measure, morphology

# Convert to grayscale if the image is RGB/RGBA
gray = color.rgb2gray(img) if img.ndim == 3 else img

# Otsu threshold (use '<' instead of '>' if objects are dark on a light background),
# remove small noise, then label connected components
binary = morphology.remove_small_objects(gray > filters.threshold_otsu(gray), min_size=50)
count = int(np.max(measure.label(binary)))

print(f"Objects counted: {count}")

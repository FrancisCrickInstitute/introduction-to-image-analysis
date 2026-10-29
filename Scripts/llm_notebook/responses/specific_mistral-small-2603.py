import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure
import pandas as pd

# Assuming 'img' is already loaded in your notebook as a numpy array
# with shape (height, width, 4) where channels are [Nuclei, Tubulin, Actin, YAP/TAZ]

# Extract the nuclear channel (channel 0)
nuclear_channel = img[:, :, 0]

# Simple thresholding using Otsu's method (automatic threshold selection)
threshold_value = filters.threshold_otsu(nuclear_channel)
binary_mask = nuclear_channel > threshold_value

# Label connected regions (cells)
labeled_cells = measure.label(binary_mask)
num_cells = np.max(labeled_cells)

# Create an overlay image (original RGB with segmented nuclei in red)
overlay = img.copy()
overlay[labeled_cells > 0] = [255, 0, 0, 0]  # Set segmented cells to red (RGBA)

# Display the overlay
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(nuclear_channel, cmap='gray')
plt.title('Nuclear Channel')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title(f'Segmented Cells (n={num_cells})')
plt.axis('off')
plt.show()

# Create results table
results = pd.DataFrame({
    'Input filename': [filename],
    'Thresholding method used': ['Otsu'],
    'Thresholding value used': [threshold_value],
    'Number of cells': [num_cells],
    'Notes': ['Simple Otsu thresholding on nuclear channel']
})

# Display table in markdown format
display(results.style.hide(axis="index"))

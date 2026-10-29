import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

# Assuming img is a 3D numpy array with shape (height, width, channels)
# Channel order: 0=Nuclei, 1=Tubulin, 2=Actin, 3=YAP/TAZ

# Extract the nuclear channel (Hoechst)
nuclei_channel = img[:, :, 0]

# Apply Otsu's thresholding to segment nuclei
threshold_value = threshold_otsu(nuclei_channel)
binary_mask = nuclei_channel > threshold_value

# Label connected regions (cells)
labeled_cells = label(binary_mask)
num_cells = labeled_cells.max()  # Number of unique regions (cells)

# Create overlay visualization
plt.figure(figsize=(10, 8))
plt.imshow(img.max(axis=2), cmap='gray')  # Show maximum projection of all channels
plt.imshow(labeled_cells, cmap='jet', alpha=0.3)  # Overlay cell labels
plt.title(f'Cell Segmentation (n={num_cells})')
plt.axis('off')
plt.show()

# Generate markdown table output
markdown_output = f"""
| Input filename | Thresholding method | Thresholding value | Number of cells | Notes |
|----------------|--------------------|--------------------|-----------------|-------|
| {filename}     | Otsu's threshold   | {threshold_value:.2f} | {num_cells} | Simple grey-level thresholding on nuclear channel |
"""

print(markdown_output)

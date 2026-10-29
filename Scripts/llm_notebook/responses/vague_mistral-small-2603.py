import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage as ndi

# Assuming `img` is already loaded as a 2D multichannel array
# Example: img.shape = (height, width, channels)

# Step 1: Select the channel with the best contrast (e.g., channel 0)
channel = img[:, :, 0]  # Adjust channel index as needed

# Step 2: Preprocess the image
# Enhance contrast (e.g., using CLAHE)
from skimage.exposure import equalize_adapthist
channel_eq = equalize_adapthist(channel, kernel_size=None, clip_limit=0.03)

# Step 3: Denoise (e.g., using Gaussian blur)
channel_blur = filters.gaussian(channel_eq, sigma=1)

# Step 4: Thresholding (Otsu's method)
threshold = filters.threshold_otsu(channel_blur)
binary = channel_blur > threshold

# Step 5: Clean up the binary image
# Remove small objects
binary_clean = morphology.remove_small_objects(binary, min_size=100)
binary_clean = morphology.remove_small_holes(binary_clean, area_threshold=100)

# Step 6: Separate touching cells (Watershed)
# Compute distance to background
distance = ndi.distance_transform_edt(binary_clean)
local_max = morphology.peak_local_max(distance, min_distance=10, labels=binary_clean)
markers = measure.label(local_max)
labels = segmentation.watershed(-distance, markers, mask=binary_clean)

# Step 7: Count cells
num_cells = len(np.unique(labels)) - 1  # Subtract 1 for background (label 0)

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(channel, cmap='gray')
axes[0, 0].set_title('Original Channel')
axes[0, 1].imshow(channel_eq, cmap='gray')
axes[0, 1].set_title('Contrast Enhanced')
axes[0, 2].imshow(binary, cmap='gray')
axes[0, 2].set_title('Thresholded')
axes[1, 0].imshow(binary_clean, cmap='gray')
axes[1, 0].set_title('Cleaned Binary')
axes[1, 1].imshow(labels, cmap='nipy_spectral')
axes[1, 1].set_title('Segmented Cells')
axes[1, 2].text(0.5, 0.5, f'Number of Cells: {num_cells}',
               fontsize=14, ha='center')
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

print(f"Number of cells: {num_cells}")

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage

def count_cells(img, channel=0, threshold_method='otsu', min_size=10, max_size=None,
                show_steps=False, save_figures=False):
    """
    Count cells in a 2D multichannel fluorescence microscopy image.

    Parameters:
    - img: Input image (numpy array)
    - channel: Which channel to use for segmentation (default: 0)
    - threshold_method: Thresholding method ('otsu', 'yen', 'li', 'mean', 'minimum', 'triangle')
    - min_size: Minimum object size to consider as a cell (pixels)
    - max_size: Maximum object size to consider as a cell (pixels)
    - show_steps: Whether to display intermediate processing steps
    - save_figures: Whether to save intermediate figures

    Returns:
    - labeled_cells: Labeled image of identified cells
    - cell_count: Total number of cells
    """

    # Select the specified channel (assuming img is a 3D array: height x width x channels)
    if len(img.shape) == 3:
        img_channel = img[:, :, channel]
    else:
        img_channel = img

    # Display original image
    if show_steps or save_figures:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_channel, cmap='gray')
        plt.title('Original Image (Channel {})'.format(channel))
        plt.axis('off')

    # Apply background subtraction (simple method)
    background = filters.rank.mean(img_channel, morphology.disk(50))
    img_subtracted = img_channel - background

    # Apply thresholding
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(img_subtracted)
    elif threshold_method == 'yen':
        threshold = filters.threshold_yen(img_subtracted)
    elif threshold_method == 'li':
        threshold = filters.threshold_li(img_subtracted)
    elif threshold_method == 'mean':
        threshold = filters.threshold_mean(img_subtracted)
    elif threshold_method == 'minimum':
        threshold = filters.threshold_minimum(img_subtracted)
    elif threshold_method == 'triangle':
        threshold = filters.threshold_triangle(img_subtracted)
    else:
        raise ValueError("Invalid threshold method")

    binary = img_subtracted > threshold

    # Apply morphological operations to clean up the image
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = morphology.binary_opening(binary, morphology.disk(2))

    # Remove small objects
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    # Remove large objects if max_size is specified
    if max_size is not None:
        binary = morphology.remove_small_objects(~binary, min_size=max_size)
        binary = ~binary

    # Label connected components
    labeled_cells, num_cells = ndimage.label(binary)

    # Display processing steps if requested
    if show_steps or save_figures:
        plt.subplot(1, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Image (Threshold: {:.2f})'.format(threshold))
        plt.axis('off')

        if show_steps:
            plt.show()

        if save_figures:
            plt.savefig('cell_segmentation_steps.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Display labeled cells
    if show_steps or save_figures:
        plt.figure(figsize=(10, 5))
        plt.imshow(labeled_cells, cmap='nipy_spectral')
        plt.title('Labeled Cells (Total: {})'.format(num_cells))
        plt.axis('off')

        if show_steps:
            plt.show()

        if save_figures:
            plt.savefig('labeled_cells.png', dpi=300, bbox_inches='tight')
            plt.close()

    return labeled_cells, num_cells

# Example usage:
# Assuming your image is loaded into 'img' variable
# You can adjust parameters as needed
labeled_cells, cell_count = count_cells(
    img,
    channel=0,  # Use the first channel (adjust if needed)
    threshold_method='otsu',  # Try different methods if needed
    min_size=20,  # Minimum cell size in pixels
    max_size=None,  # Set if you want to exclude very large objects
    show_steps=True,  # Set to False to hide intermediate steps
    save_figures=False  # Set to True to save figures
)

print(f"Total number of cells detected: {cell_count}")

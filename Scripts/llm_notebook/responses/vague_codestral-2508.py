from skimage.measure import label
from skimage.morphology import closing, square
import numpy as np

# Assuming 'img' is a binary or grayscale image (adjust threshold if needed)
binary_img = img > np.percentile(img, 90)  # Simple thresholding
cleaned_img = closing(binary_img, square(3))  # Remove small holes
labeled_img, num_objects = label(cleaned_img, return_num=True, connectivity=2)

print(f"Number of objects: {num_objects}")

I have a 2D multichannel fluorescence microscopy image of some cells that I need to analyse. The image, in TIF format,
consists of four channels:
1. Nuclei (Hoechst)
2. Tubulin
3. Actin (Phalloidin)
4. YAP/TAZ
Can you write a Python script to determine this? The script should use the nuclear channel to segment and count the
cells, using simple grey-level thresholding. The outputs should consist of an overlay of the segmented cells on the 
original image and a table in markdown format containing the following columns:
* Input filename (stored in the variable `filename`)
* Thresholding method used
* Thresholding value used
* Number of cells
* Notes (any additional information that may be useful)
Keep the code as short and as simple as possible and make extensive use of comments to aid in interpretation.
The code will go into an existing Jupyter notebook that loads the image into a variable called `img`.

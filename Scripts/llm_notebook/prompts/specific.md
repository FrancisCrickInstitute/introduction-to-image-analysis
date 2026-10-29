I have a 2D multichannel fluorescence microscopy image of some cells that I need to analyse. The image, in TIF format,
consists of 4 channels:
1.	Nuclei (Hoechst)
2.	Tubulin
3.	Actin (Phalloidin)
4.	YAP/TAZ
Can you write a python script to determine this? The script should use the nuclear channel to segment and count the
cells, using simple grey-level thresholding. The outputs should consist of an overlay of the segmented cells on the 
original image, and a table containing the following columns:
•	Input filename
•	Number of cells
Keep the code as short and as simple as possible and make extensive use of comments to aid in interpretation.

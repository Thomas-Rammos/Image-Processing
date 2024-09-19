# Image-Processing
Design and implementation of various image processing tasks using Python.

# Introduction
This project is part of the MΥΕ037 course on Digital Image Processing. The goal of the project is to implement several key algorithms for image processing tasks such as image patch extraction, convolution, edge detection, and applying filters like Gaussian and Sobel. The project is written in Python using libraries such as NumPy and Matplotlib.

# How it works
The project includes a Python script filters.py that contains the implementation of various image processing functions. The following tasks are implemented:

- Image patches: Divides an input grayscale image into non-overlapping 16x16 patches and normalizes each patch.
- Convolution: Performs convolution of an image with a specified kernel, applying zero-padding.
- Edge detection: Detects edges in an image by computing the gradient magnitude.
- Sobel operator: Applies the Sobel filter to detect horizontal and vertical gradients.
- Laplacian of Gaussian (LoG): Applies LoG filters to detect edges and other features.
The project includes a main function that loads a test image (grace_hopper.png), applies the implemented functions, and saves the results to specified directories.
# How to use
  1. Clone the repository and ensure that the required dependencies (NumPy, SciPy, Matplotlib) are installed.
    git clone <repository-url>
    cd <repository-folder>
    pip install -r requirements.txt


  2. To run the image processing tasks, execute the script filters.py. The script will generate and save the processed images in corresponding directories.
    python filters.py

  3. For each task:
    - The image patches are saved in the image_patches/ folder.
    - The results of convolution and Gaussian filtering are saved in gaussian_filter/.
    - Sobel operator results are saved in sobel_operator/.
    - LoG filter results are saved in log_filter/.
  4. Open the generated folders to visualize the results or use Matplotlib to display the images directly within the script.
  5. The report for the project (assignment.pdf) contains detailed theoretical explanations and plots of the results.


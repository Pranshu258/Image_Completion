# Statistics of Patch Offsets for Image Completion
A Python Implementation: [Pranshu Gupta](https://github.com/Pranshu258/) and [Shrija Mishra](https://github.com/shrija14/)

### Introduction
Content Aware Image Completion is the task of filling the missing part of an image with content like the rest of the image. We have implemented an application that would take an input image and a mask denoting the missing part and give a complete image as the final artifact. One of the best usage of this application can be object removal from images.

### Scope
Replicate the results in the paper “Statistics of Patch Offsets for Image Completion” by Kaiming He and Jian Sun [1]. This involves the following implementations:
- Finding similar patches and obtaining their offsets by using the algorithm described in the paper “Computing Nearest Neighbor - Fields via Propagation-Assisted KD-Trees” by Kaiming He and Jian Sun [2].
- Finding K dominant offsets through computation of statistics by a 2-D Histogram.
- Finding the optimal labelling for the unknown pixels by the approach proposed in the paper “Fast Approximate Energy Minimization via Graph Cuts” by Boykov, Veksler and Zabih [3].
- Completing the image based on the labels found in the previous step.

### Usage Instructions
- Python Version – 2.7.4
- OpenCV Version – 3.4.3
- PyMaxflow must be installed in the system (pip install PyMaxflow)
- Keep the image to be completed and the mask in images/source folder.
- In command/anaconda prompt run the application in the following way 
```python main.py <image file name> <mask file name>``` e.g. ```python main.py brick.jpg brick.png```. This will run the code and the output will be in images/output folder.

### Resources
1. He, Kaiming, and Jian Sun.: Statistics of patch offsets for image completion. ECCV (2012) 16-29.
2. He, K., Sun, J.: Computing nearest-neighbor fields via propagation-assisted kdtrees. CVPR (2012)
3. Boykov, Y., Veksler, O., Zabih, R., Fast approximate energy minimization via graph cuts. TPAMI (2001) 1222 - 1239
4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
5.  https://github.com/scipy/scipy/blob/v1.1.0/scipy/spatial/kdtree.py
6. http://pmneila.github.io/PyMaxflow/maxflow.html
7.  https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp
8. https://github.com/nsubtil/gco-v3.0/blob/master/GCO_README.TXT
9. http://peekaboo-vision.blogspot.com/2012/05/graphcuts-for-python-pygco.html

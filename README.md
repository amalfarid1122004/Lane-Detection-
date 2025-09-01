# Lane-Detection
Developed a computer vision–based Lane Detection System to support Advanced Driver Assistance Systems (ADAS) and autonomous driving applications. The system processes video frames from a forward-facing camera to detect and highlight road lane markings in real time.

Using OpenCV and NumPy, the pipeline includes image preprocessing (grayscale conversion, Gaussian blur), Canny edge detection, region of interest masking, and Hough Transform for line detection. Polynomial fitting and perspective transformation were applied to handle curves and enhance lane visibility under varying road and lighting conditions.

The system outputs an overlay of detected lanes on the driving scene, along with lane curvature and vehicle deviation reports, enabling safer navigation and driver assistance.

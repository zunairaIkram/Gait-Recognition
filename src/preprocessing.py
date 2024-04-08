import cv2
import numpy as np

def histogram_equalization(image):
    if len(image.shape) == 2:  # If a Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image
        # Convert to YUV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Apply histogram equalization on the Y channel
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

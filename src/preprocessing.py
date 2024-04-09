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

def image_threshold(image):
    # Convert the image to grayscale
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply background subtraction (optional)
    # foreground_mask = apply_background_subtraction(gray)
    _, thresholded_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image


#not being used yet
def detect_edges(image): #with otsu(for automatic selecting threshold value)
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Perform edge detection
    edges = cv2.Canny(binary, 0.5 * cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0],
                      cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return edges



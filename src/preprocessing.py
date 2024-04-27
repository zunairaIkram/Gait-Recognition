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


#not being used yet, for simplicity use edges = cv2.Canny(gray, 100, 255) if required
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


def extract_binary_mask(image):
    # Convert image to grayscale
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply thresholding to obtain binary mask
    _, binary_mask1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    binary_mask = cv2.bitwise_and(binary_mask1, binary_mask2)
    return binary_mask

def silhoutte_extract(image, binary_mask):
        if len(image.shape) != 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create an empty mask for the silhouette
        mask = np.zeros_like(gray)
        # Draw contours on the silhouette mask
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        # Apply morphological operations to refine the silhouette
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.bitwise_and(image, image, mask=mask)     
        return mask   

     


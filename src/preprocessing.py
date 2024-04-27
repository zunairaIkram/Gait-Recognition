import cv2
import numpy as np

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

# Function to calculate standard deviation of an image
def calculate_std_dev(image):
    if len(image.shape) == 3:  # Color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    return np.std(gray_image)

def histogram_equalization(image):
    if len(image.shape) == 2:  # If a Grayscale image
        image_gray = cv2.equalizeHist(image_gray)
    else:  # RGB image
        # Convert image from RGB to HSV
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Histogram equalisation on the V-channel
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        # Convert back to BGR
        image_RGB = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return image_RGB

def CLAHE(image, std_dev_threshold=20):                                          #Contrast Limited Adaptive Histogram Equalization
    # Check the standard deviation of the image
    if calculate_std_dev(image) < std_dev_threshold:
        # Image needs contrast enhancement
        if len(image.shape) == 2:  # Grayscale image
            clahe = cv2.createCLAHE()
            return clahe.apply(image)
        else:  # RGB image, assumed to be in RGB format as per function description
            # Convert image from RGB to HSV (OpenCV uses BGR by default)
            img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Create a CLAHE object (with default parameters)
            clahe = cv2.createCLAHE()
            # Apply CLAHE on the V-channel
            img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
            return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        # Return the original image if standard deviation is above the threshold
        return image

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
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
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

    
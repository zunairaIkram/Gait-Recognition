import cv2
import numpy as np

def extract_silhouette(image, binary):
    # Convert the image to grayscale if it's not already
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the silhouette
    silhouette_mask = np.zeros_like(gray)

    # Draw the largest contour on the silhouette mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouette_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    return silhouette_mask

import cv2
import numpy as np

def extract_silhouette(image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask for the silhouette
    silhouette_mask = np.zeros_like(gray)
    
    # Draw contours on the silhouette mask
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw the approximated contour on the silhouette mask
        cv2.drawContours(silhouette_mask, [approx], -1, (255), thickness=cv2.FILLED)
    
    return silhouette_mask


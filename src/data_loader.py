import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from preprocessing import CLAHE,  preprocess_image
from silhouette_extraction import extract_silhouette
from segmentation_model import object_detection_api

def display_images_with_matplotlib(image_list, titles):
    # Set up the figure and axes
    fig, axes = plt.subplots(1, len(image_list), figsize=(10, 5))  # Adjust size as needed
    for ax, img, title in zip(axes, image_list, titles):
        # OpenCV reads images in BGR, convert to RGB for correct color display in Matplotlib
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')  # Turn off axis numbers and ticks
    
    plt.show()
    
def read_images(root_folder):
    frames = []
    # Iterate through each subfolder in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # Check if the current item in the root folder is a directory
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_name}")
            
            # Iterate through each subfolder in the current folder
            for sub_folder_name in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder_name)
                
                # Check if the current item is a directory
                if os.path.isdir(sub_folder_path):
                    print(f"\nProcessing subfolder: {sub_folder_name}")
                    
                    # Iterate through each image file in the current subfolder
                    for image_name in os.listdir(sub_folder_path):
                        image_path = os.path.join(sub_folder_path, image_name)
                        
                        # Check if the current item is a file and if it's an image (you can add more checks if needed)
                        if os.path.isfile(image_path) and image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            print(f"Reading image: {image_name}")
                            
                            # Read the image using OpenCV
                            image = cv2.imread(image_path)
                            # Process the image for gait recognition (e.g., feature extraction, normalization, etc.)
                            # Your gait recognition code goes here

                            if image is not None:
                                image_eq = CLAHE(image)        #Histogram
                                processed_image = preprocess_image(image_eq)
                                # silhoutte = extract_silhouette(image, processed_image)


                                segmented = object_detection_api(image)

                        

                                # Display the image
                                display_images_with_matplotlib([image, image_eq, segmented ],["Original Image", "CLAHE Enhanced", "processed_image"])

                            else:
                                print(f"Failed to load image: {image_name}")
                            
                        else:
                            print(f"Skipping non-image file: {image_name}")
                else:
                    print(f"Skipping non-directory: {sub_folder_name}")
        else:
            print(f"Skipping non-directory: {folder_name}")

    return frames
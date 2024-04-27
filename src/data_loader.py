import os
import cv2
import numpy as np
from preprocessing import CLAHE, histogram_equalization, image_threshold, detect_edges, extract_binary_mask, silhoutte_extract
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
                                image_eq = CLAHE(image)        #Histogram Equalization
                                #image_thres = image_threshold(image_eq)
                                #image_edge = detect_edges(image_thres)
                                gray = cv2.cvtColor(image_eq, cv2.COLOR_BGR2GRAY)
                                
                                
                                binary_mask = extract_binary_mask(image)
                                
                                #silhoutte_mask = silhoutte_extract(image, binary_mask)
                                
                                
                                
                                
                
                                
                                frames.append(binary_mask)
                                
                            
                                # Display the image
                                cv2.imshow('Image', binary_mask)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                            else:
                                print(f"Failed to load image: {image_name}")
                            
                        else:
                            print(f"Skipping non-image file: {image_name}")
                else:
                    print(f"Skipping non-directory: {sub_folder_name}")
        else:
            print(f"Skipping non-directory: {folder_name}")
    return frames

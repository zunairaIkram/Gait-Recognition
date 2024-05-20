from data_loader import read_images 
import os

# Define the path to the root folder containing subfolders with images
root_folder =  r"..\data\train\images"

silhouette_folder = '../data/train/silhouette/'
if not os.path.exists(silhouette_folder):
    os.makedirs(silhouette_folder)

# Call the function to start processing images
read_images(root_folder, silhouette_folder)


#  for frame in frames:
#     print(f"{frames[i]}\n")
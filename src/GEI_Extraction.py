# Gait Energy Image
import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
# Define the base directories

# Function to check if a file is an image
def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

# Function to calculate the mass center of an image
def mass_center(img,is_round=True):
    Y = img.mean(axis=1)
    X = img.mean(axis=0)
    Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
    X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
    if is_round:
        return int(round(X_)),int(round(Y_))
    return X_,Y_

def image_extract(img,newsize):
   
    print("Processed image dimensions:", img.shape)
    
    x_s_arr = np.where(img.mean(axis=0) != 0)[0]
    y_s_arr = np.where(img.mean(axis=1) != 0)[0]
    
    if len(x_s_arr) == 0 or len(y_s_arr) == 0:
        print("Error: No nonzero values found along the specified axis.")
        return None
    
    x_s = x_s_arr.min()
    x_e = x_s_arr.max()
    y_s = y_s_arr.min()
    y_e = y_s_arr.max()
    
    img = img[y_s:y_e+1, x_s:x_e+1]  # Ensure that the full range is included
    img_resized = resize(img, newsize)
    
    return img_resized

# Function to create GEI for each person and scene
def create_gei(segmentation_base_dir, gei_base_dir, newsize=(128, 64)):
    if not os.path.exists(gei_base_dir):
        os.makedirs(gei_base_dir)
    gei_results = {}
    for person_id in os.listdir(segmentation_base_dir):
        person_dir = os.path.join(segmentation_base_dir, person_id)
        if os.path.isdir(person_dir):
            person_gei_dir = os.path.join(gei_base_dir, person_id)
            if not os.path.exists(person_gei_dir):
                os.makedirs(person_gei_dir)

            for scene_id in os.listdir(person_dir):
                scene_dir = os.path.join(person_dir, scene_id)
                if os.path.isdir(scene_dir):
                    images = []
                    for file in os.listdir(scene_dir):
                        if is_image_file(file):
                            filename = os.path.join(scene_dir, file)
                            try:
                                print(filename)
                                image = imread(filename)
                                image = image_extract(image, newsize)
                                if image is not None:
                                    images.append(image)
                                else:
                                    print("Error: Image extraction failed for", filename)
                            except Exception as e:
                                print(f"Error reading {filename}: {e}")

                    if images:
                        gei = np.mean(images, axis=0)
                        print(f"gei shape: {gei.shape}")
                        gei_filename = os.path.join(person_gei_dir, f"{scene_id}.jpg")
                        plt.imsave(gei_filename, gei, cmap='gray')
                        gei_results[(person_id, scene_id)] = gei
                        # print(f"GEI saved for {scene_id} at {gei_filename}")

    return gei_results

print("GEI generation complete.")
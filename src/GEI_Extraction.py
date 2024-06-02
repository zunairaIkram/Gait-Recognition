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
import os
import numpy as np
import matplotlib.pyplot as plt

def create_gei(silhouette_data, gei_base_dir, newsize=(128, 64)):
    if not os.path.exists(gei_base_dir):
        os.makedirs(gei_base_dir)

    gei_results = {}

    current_person_id = None
    current_sub_folder = None
    images = []

    def process_images(person_id, sub_folder_name, images):
        if images:
            gei = np.mean(images, axis=0)
            print(f"gei shape: {gei.shape}")
            person_gei_dir = os.path.join(gei_base_dir, person_id)
            if not os.path.exists(person_gei_dir):
                os.makedirs(person_gei_dir)
            gei_filename = os.path.join(person_gei_dir, f"{sub_folder_name}.jpg")
            plt.imsave(gei_filename, gei, cmap='gray')
            gei_results[(person_id, sub_folder_name)] = gei

    for person_id, sub_folder_name, image_name, thresholded_image in silhouette_data:
        if person_id != current_person_id:
            # Process the remaining images of the previous person
            process_images(current_person_id, current_sub_folder, images)
            current_person_id = person_id
            current_sub_folder = None
            images = []

        if sub_folder_name != current_sub_folder:
            # Process the previous subfolder's images
            process_images(current_person_id, current_sub_folder, images)
            current_sub_folder = sub_folder_name
            images = []

        try:
            print(f"Processing {image_name}")
            image = thresholded_image
            image = image_extract(image, newsize)

            if image is not None:
                images.append(image)
                print(len(images))
            else:
                print("Error: Image extraction failed for", person_id, sub_folder_name, image_name)
        except Exception as e:
            print(f"Error reading {image_name}: {e}")

    # Process the last subfolder's images of the last person
    process_images(current_person_id, current_sub_folder, images)

    print("GEI generation complete.")
    return gei_results


# Save GEI data in pkl file
def save_gei_data(gei_results):
    X = []
    y = []
    for (person_id, _), gei in gei_results.items():
        X.append(gei)
        y.append(person_id)

    data = {'X': X, 'y': y}
    return data
    # with open(output_file, 'wb') as f:
    #     pickle.dump(data, f)
    # print(f"GEI data saved to {output_file}")

# def create_gei(silhouette_data, gei_base_dir, newsize=(128, 64)):
#     if not os.path.exists(gei_base_dir):
#         os.makedirs(gei_base_dir)
    
#     gei_results = []
#     for person_id, scenes in silhouette_data.items():
#         person_gei_dir = os.path.join(gei_base_dir, person_id)
#         if not os.path.exists(person_gei_dir):
#             os.makedirs(person_gei_dir)

#         for scene_id, image_paths in scenes.items():
#             images = []
#             for image_path in image_paths:
#                 try:
#                     print(image_path)
#                     image = imread(image_path)
#                     image = image_extract(image, newsize)
#                     if image is not None:
#                         images.append(image)
#                     else:
#                         print("Error: Image extraction failed for", image_path)
#                 except Exception as e:
#                     print(f"Error reading {image_path}: {e}")

#             if images:
#                 gei = np.mean(images, axis=0)
#                 print(f"gei shape: {gei.shape}")
#                 gei_filename = os.path.join(person_gei_dir, f"{scene_id}.jpg")
#                 plt.imsave(gei_filename, gei, cmap='gray')
#                 gei_results[(person_id , scene_id)]= gei

#     print("GEI generation complete.")
#     return gei_results
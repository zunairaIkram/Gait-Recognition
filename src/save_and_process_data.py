import os
import cv2
import pickle
from silhouette_extraction import get_silhouette
from GEI_Extraction import create_gei

def save_pkl(data, filename):
    try:
        full_path = os.path.abspath(filename)
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {full_path}.")
    except Exception as e:
        print(f"Failed to save data to {filename}: {e}")

def save_dir(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    if cv2.imwrite(image_path, image):
        print(f"Successfully saved: {filename} to {path}")
    else:
        print(f"Failed to save: {filename} to {path}.")

def load_images(root_folder):
    data = []
    # Iterate through each subfolder in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for sub_folder_name in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder_name)
                if os.path.isdir(sub_folder_path):
                    image_files = os.listdir(sub_folder_path)
                    total_images = len(image_files)
                    images_processed = 0
                    for image_name in image_files:
                        images_processed += 1
                        image_path = os.path.join(sub_folder_path, image_name)
                        if os.path.isfile(image_path) and image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            print(f"Loading image: {image_name} ({images_processed}/{total_images}) in {sub_folder_path}")
                            image = cv2.imread(image_path)
                            if image is not None:
                                data.append((folder_name, sub_folder_name, image_name, image))
                            else:
                                print(f"Failed to load image: {image_name}")
                        else:
                            print(f"Skipping non-image file: {image_name} ({images_processed}/{total_images}) in {sub_folder_path}")
    return data



def process_and_save_silhouettes(data, silhouette_folder, silhouette_data_file):
    silhouette_data = []
    total_images = len(data)
    for idx, item in enumerate(data, start=1):
        if len(item) == 4:
            folder_name, sub_folder_name, image_name, image = item
            silhouette_path = os.path.join(silhouette_folder, folder_name, sub_folder_name, image_name)
            
            # Check if the silhouette already exists
            if os.path.exists(silhouette_path):
                print(f"Silhouette already exists: {silhouette_path}. Loading instead of processing.")
                thresholded_image = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Processing silhouette {idx}/{total_images}: {image_name} in {folder_name}/{sub_folder_name}")
                silhouette = get_silhouette(image)
                _, thresholded_image = cv2.threshold(silhouette, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                save_dir(thresholded_image, os.path.join(silhouette_folder, folder_name, sub_folder_name), image_name)

            silhouette_data.append((folder_name, sub_folder_name, image_name, thresholded_image))
        else:
            print(f"Skipping invalid data item at index {idx}.")

    save_pkl(silhouette_data, silhouette_data_file)
    print("All silhouette data has been saved to the pickle file.")


def process_and_save_geis(silhouette_data, gei_folder, gei_data_file, newsize=(128, 64)):
    gei_results = create_gei(silhouette_data, gei_folder, newsize)
    gei_data = []
    total_geis = len(gei_results)
    for idx, ((person_id, scene_id), gei) in enumerate(gei_results.items(), start=1):
        print(f"Processing GEI {idx}/{total_geis}: {person_id} scene {scene_id}")
        gei_data.append((person_id, scene_id, gei))
    save_pkl(gei_data, gei_data_file)
    print("All GEI data has been saved to the pickle file.")

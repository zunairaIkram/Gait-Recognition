import os
from save_and_process_data import load_images, save_pkl, process_and_save_silhouettes, process_and_save_geis
from data_loader import load_data_pkl

# Define paths and directories
root_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\images"
current_directory = os.getcwd()
data_file = os.path.join(current_directory, 'image_data.pkl')
silhouette_data_file = os.path.join(current_directory, 'silhouette_data.pkl')
silhouette_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\silhouette"
gei_data_file = os.path.join(current_directory, 'gei_data.pkl')
gei_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\gei"

# Load image data and process silhouettes
if not os.path.exists(data_file):
    print("Image data not found in pickle file, loading from directory...")
    image_data = load_images(root_folder)
    save_pkl(image_data, data_file)
    print("Silhouette data not found in pickle file, processing from image data...")
    silhouette_data = process_and_save_silhouettes(image_data, silhouette_folder, silhouette_data_file)
else:
    print("Loading image data from existing pickle file...")
    image_data = load_data_pkl(data_file)
    if not os.path.exists(silhouette_data_file):
        print("Silhouette data not found in pickle file, processing from image data...")
        silhouette_data = process_and_save_silhouettes(image_data, silhouette_folder, silhouette_data_file)
    else:
        print("Loading silhouette data from existing pickle file...")
        silhouette_data = load_data_pkl(silhouette_data_file)

# Process GEI data
if not os.path.exists(gei_data_file):
    print("GEI data not found in pickle file, processing from silhouette data...")
    gei_data = process_and_save_geis(silhouette_folder, gei_folder, gei_data_file)
else:
    print("Loading GEI data from existing pickle file...")
    gei_data = load_data_pkl(gei_data_file)
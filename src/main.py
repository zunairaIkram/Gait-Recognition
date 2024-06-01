import os
from save_and_process_data import load_images, save_pkl, process_and_save_silhouettes, process_and_save_geis
from data_loader import load_data_pkl
from gaitRecognitionModel import gait_recognition
from prediction import load_model_and_predict

# Define paths and directories
root_folder = r"C:\Users\Fakhi\Desktop\dipProject\train\New folder\images"
current_directory = os.getcwd()
data_file = r"C:\Users\Fakhi\Desktop\dipProject\train\image_data.pkl"
silhouette_data_file = r"C:\Users\Fakhi\Desktop\dipProject\train\silhouette_data.pkl"
silhouette_folder = r"C:\Users\Fakhi\Desktop\dipProject\train\silhouette"
gei_data_file = r"C:\Users\Fakhi\Desktop\dipProject\train\gei_data.pkl"
gei_folder = r"C:\Users\Fakhi\Desktop\dipProject\train\New folder\gei"

def load_data(path):
    try:
        if os.path.exists(data_file):
            image_data = load_data_pkl(data_file)
            return True, "Data Loaded from a pickle file."
        else:
            image_data = load_images(path)
            save_pkl(image_data, data_file)
            return True, "Data Loaded and Saved in pickle file successfully."
    except Exception as e:
        return False, f"Error loading image data: {e}"

def extract_silhouettes():
    try:
        if not os.path.exists(data_file):
            return False, "Image data file not found. Please load data first."

        if os.path.exists(silhouette_data_file):
            return True, "Silhouettes already extracted and saved to pickle file."
        else:
            image_data = load_data_pkl(data_file)
            process_and_save_silhouettes(image_data, silhouette_folder, silhouette_data_file)
            return True, "Silhouettes extracted successfully."
    except Exception as e:
        return False, f"Error extracting silhouettes: {e}"

def extract_geis():
    try:
        if not os.path.exists(silhouette_data_file):
            return False, "Silhouette data not found. Please extract silhouette data first."

        if os.path.exists(gei_data_file):
            return True, "GEIs already extracted and saved to pickle file."
        else:
            silhouette_data = load_data_pkl(silhouette_data_file)
            process_and_save_geis(silhouette_data, gei_folder, gei_data_file)
            return True, "GEIs extracted successfully."
    except Exception as e:
        return False, f"Error extracting GEIs: {e}"
    
def train_model():
    try:
        if not os.path.exists(gei_data_file):
            return False, "GEI data not found. Please extract GEI data first."
        else:
            gei_data = load_data_pkl(gei_data_file)
            acc = gait_recognition(gei_data)
            return True, f"Model trained successfully. \n Accuracy Score of Model is: {acc}"
    except Exception as e:
        return False, f"Error training model: {e}"

def predict(test_image_paths):
    try:
        if not os.path.exists('finalized_model_11_labels.sav') or not os.path.exists('pca_model_11_labels.sav'):
            return False, "Saved model not found. Please train the model first."
        
        predictions = load_model_and_predict(test_image_paths)
        results = []
        for path, prediction in zip(test_image_paths, predictions):
            # Extract the original label from the file path
            original_label = os.path.basename(os.path.dirname(path))
            results.append((path, prediction, original_label))
        
        print("Results:", results)  # Debugging line to check the output
        return True, results
    except Exception as e:
        return False, f"Error during prediction: {e}"



# import os
# from save_and_process_data import load_images, save_pkl, process_and_save_silhouettes, process_and_save_geis
# from data_loader import load_data_pkl

# # Define paths and directories
# root_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\images"
# current_directory = os.getcwd()
# data_file = os.path.join(current_directory, 'image_data.pkl')
# silhouette_data_file = os.path.join(current_directory, 'silhouette_data.pkl')
# silhouette_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\silhouette"
# gei_data_file = os.path.join(current_directory, 'gei_data.pkl')
# gei_folder = r"E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\gei"

# # Load image data and process silhouettes
# if not os.path.exists(data_file):
#     print("Image data not found in pickle file, loading from directory...")
#     image_data = load_images(root_folder)
#     save_pkl(image_data, data_file)
#     print("Silhouette data not found in pickle file, processing from image data...")
#     silhouette_data = process_and_save_silhouettes(image_data, silhouette_folder, silhouette_data_file)
# else:
#     print("Loading image data from existing pickle file...")
#     image_data = load_data_pkl(data_file)
#     if not os.path.exists(silhouette_data_file):
#         print("Silhouette data not found in pickle file, processing from image data...")
#         silhouette_data = process_and_save_silhouettes(image_data, silhouette_folder, silhouette_data_file)
#     else:
#         print("Loading silhouette data from existing pickle file...")
#         silhouette_data = load_data_pkl(silhouette_data_file)

# # Process GEI data
# if not os.path.exists(gei_data_file):
#     print("GEI data not found in pickle file, processing from silhouette data...")
#     gei_data = process_and_save_geis(silhouette_folder, gei_folder, gei_data_file)
# else:
#     print("Loading GEI data from existing pickle file...")
#     gei_data = load_data_pkl(gei_data_file)

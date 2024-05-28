import numpy as np
from PIL import Image
import pickle

def resize_image(image, target_size):
    """Resize an image to the target size using high-quality resampling."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def preprocess_image(file_path, target_size):
    """Process an image to make it compatible with the model."""
    img = Image.open(file_path).convert('L')
    img = resize_image(img, target_size)
    bw = np.array(img.point(lambda x: 0 if x < 128 else 255), dtype=np.float32)
    return bw.flatten()  # Flatten to make it compatible with model input

# Load models from disk
lg_model = pickle.load(open('finalized_model_10_labels.sav', 'rb'))
pca_model = pickle.load(open('pca_model_4_labels.sav', 'rb'))

# Path to the new image
# new_image_path = r'E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\segmentations\002\scene1_bg_L_090_1\015.jpg'

# Preprocess the image
processed_image = preprocess_image(new_image_path, (200, 250))
processed_image = processed_image.reshape(1, -1)  # Reshape for a single sample

# Apply PCA
processed_image = pca_model.transform(processed_image)

# Make a prediction
predicted_label = lg_model.predict(processed_image)
print(f"Predicted label: {predicted_label}")
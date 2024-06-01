import numpy as np
from PIL import Image
import pickle

def load_model_and_predict(image_paths):
    # Load the saved models
    lg = pickle.load(open('finalized_model_11_labels.sav', 'rb'))
    pca = pickle.load(open('pca_model_11_labels.sav', 'rb'))

    # Preprocess the images
    images = []
    """Process an image to make it compatible with the model."""
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        # img = resize_image(img, target_size)
        # image = image.resize((64, 128))  # Resize to the same size as training images
        # bw = np.array(image.point(lambda x: 0 if x < 128 else 255), dtype=np.float32)
        # images.append(bw.flatten())
        image_array = np.array(image, dtype=np.float32)
        images.append(image_array.flatten())
        
    # Convert the list of images to a numpy array
    images = np.array(images)

    # Transform the images using PCA
    images_pca = pca.transform(images)

    # Predict the labels
    predictions = lg.predict(images_pca)

    # print(f"Predicted label: {predictions}")

    return predictions
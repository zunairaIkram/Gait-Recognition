import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pickle
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# def visualize_results(y_true, y_pred):
#     """Visualize the results of gait recognition."""
#     plt.figure(figsize=(10, 6))
#     sns.set(style="whitegrid")
    
#     # Plot the distribution of true labels
#     plt.subplot(1, 2, 1)
#     sns.countplot(y_true)
#     plt.title('Distribution of True Labels')
#     plt.xlabel('True Label')
#     plt.ylabel('Count')
    
#     # Plot the distribution of predicted labels
#     plt.subplot(1, 2, 2)
#     sns.countplot(y_pred)
#     plt.title('Distribution of Predicted Labels')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('Count')
    
#     plt.tight_layout()
#     plt.show()

target_size = (64, 128)
def resize_image(image, target_size):
    """Resize an image to the target size using high-quality resampling."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def gait_recognition(data):
    # Data preprocessing
    # X = []
    # y = []

    # Extract X and y from the loaded data
    X, y = data['X'], data['y']

    # Ensure that data is a dictionary
    if isinstance(data, dict) and 'X' in data and 'y' in data:
        X, y = data['X'], data['y']
    else:
        raise ValueError("Loaded data is not in the expected format: {'X': ..., 'y': ...}")
    
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Determine the size of the images
    if len(X.shape) == 4:  # Assuming the shape is (num_samples, height, width, channels)
        _, height, width, channels = X.shape
    elif len(X.shape) == 3:  # Assuming the shape is (num_samples, height, width)
        _, height, width = X.shape
        channels = 1
    else:
        raise ValueError("Unexpected shape of X. Expected 3 or 4 dimensions.")

    print(f"Training images size: {height}x{width} with {channels} channels")

    # PCA Transformation
    X = X.reshape(X.shape[0], -1)
    pca = PCA(0.99)
    # X_pca = pca.fit_transform(X)
    

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # print(X_train_pca[0], y_train[0])
    # Logistic Regression Training
    # lg = LogisticRegression(solver ='lbfgs',multi_class='multinomial')
    # lg = lg.fit(X_train_pca, y_train)
    # y_pred = lg.predict(X_test_pca)
    # acc = accuracy_score(y_test, y_pred)
    C_values = [1, 0.8, 0.3, 0.1, 0.03, 0.001, 0.0001]
    best_C = None
    best_accuracy = 0
    accuracies = []

    for C in C_values:
        svm = LinearSVC(C=C)
        svm.fit(X_train_pca, y_train)
        y_pred = svm.predict(X_test_pca)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append((C, acc))
        
        print(f"Linear SVC with C={C}")
        print(f'Accuracy: {acc}')
        print("-----------------------------------------------------------------------")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_C = C

    print(f'Best C value: {best_C} with accuracy: {best_accuracy}')

    # Save the best model
    best_svm = LinearSVC(C=best_C)
    best_svm.fit(X_train_pca, y_train)
    
    

    
    # sns.countplot(y)
    # plt.show()

    # Save the models to disk
    pickle.dump(best_svm, open('finalized_model_11_labels.sav', 'wb'))
    pickle.dump(pca, open('pca_model_11_labels.sav', 'wb'))
    # visualize_results(y_test, y_pred)
    return acc

        # if isinstance(data, dict):
    #         print("Data keys:", data.keys())
    # elif isinstance(data, list):
    #     print("First item type in list:", type(data[0]))
    #     if isinstance(data[0], dict):
    #         print("Keys in first dictionary item:", data[0].keys())
    #     elif isinstance(data[0], (list, tuple)):
    #         print("Length of first list/tuple item:", len(data[0]))
    # else:
    #     print("Loaded data structure is not recognized")
    # data = r'E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\segmentations'
    # target_size = (200, 250)  # Correct order (width, height)

    # def resize_image(image, target_size):
    #     """Resize an image to the target size using high-quality resampling."""
    #     return image.resize(target_size, Image.Resampling.LANCZOS)

    # limit = 11
    # # limit_count = 0
    # for sub_dir in tqdm(os.listdir(data), desc="Processing directories"):
        
    #     # if limit_count == limit:
    #     #     break
    #     # limit_count += 1

    #     label = int(sub_dir[1:])
    #     for sub_dir2 in os.listdir(os.path.join(data, sub_dir)):
    #         # image1 = np.zeros(target_size[::-1], dtype=np.float32)  # Note: np.zeros uses (height, width)
    #         for filename in os.listdir(os.path.join(data, sub_dir, sub_dir2)):
    #             if filename.endswith('.jpg'):
    #                 file_path = os.path.join(data, sub_dir, sub_dir2, filename)
    #                 img = Image.open(file_path).convert('L')
    #                 img = resize_image(img, target_size)
                    
    #                 bw = np.array(img.point(lambda x: 0 if x < 128 else 255), dtype=np.float32)
                    
    #                 X.append(bw)
    #                 y.append(label)
    
    # # Save the arrays
    # with open('X_10_data.pkl', 'wb') as f:
    #     pickle.dump(X, f)
    # with open('y_10_data.pkl', 'wb') as f:
    #     pickle.dump(y, f)

    # # Load the arrays
    # with open('X_7_data.pkl', 'rb') as f:
    #     X = pickle.load(f)
    # with open('y_7_data.pkl', 'rb') as f:
    #     y = pickle.load(f)

    # print("data loaded")

# # Define different models and solvers
# models = {
#     'LogisticRegression': {
#         'model': LogisticRegression,
#         'params': {'multi_class': 'multinomial', 'solver': 'lbfgs', 'max_iter': 1000},
#         'polynomial': True
#     },
#     'SVC': {
#         'model': SVC,
#         'params': {},
#         'polynomial': True
#     },
#     'RandomForest': {
#         'model': RandomForestClassifier,
#         'params': {'n_estimators': 100},
#         'polynomial': False
#     }
# }

# results = []

# for name, info in models.items():
#     model = info['model'](**info['params'])
#     if info['polynomial']:
#         poly = PolynomialFeatures(degree=2)
#         x_train_poly = poly.fit_transform(x_train)
#         x_test_poly = poly.transform(x_test)
#     else:
#         x_train_poly, x_test_poly = x_train, x_test
    
#     model.fit(x_train_poly, y_train)
#     y_pred = model.predict(x_test_poly)
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')

#     results.append({
#         'Model': name,
#         'Accuracy': acc,
#         'F1 Score': f1
#     })
    
#     # Save each model
#     model_filename = f'finalized_model_{name}.sav'
#     pca_filename = f'pca_model_{name}.sav'
#     pickle.dump(model, open(model_filename, 'wb'))
#     pickle.dump(pca, open(pca_filename, 'wb'))

# # Convert results to DataFrame and save as Excel
# df_results = pd.DataFrame(results)
# df_results.to_excel('model_performance.xlsx', index=False)
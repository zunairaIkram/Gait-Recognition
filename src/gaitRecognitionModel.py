import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def gait_recognition(base_path):
    # Data preprocessing
    X = []
    y = []
    # base_path = r'E:\Zunaira\UniversityCourses\Semester5\DIP\PROJECT(Human Gait Recognition)\Development\data\train\segmentations'
    target_size = (200, 250)  # Correct order (width, height)

    def resize_image(image, target_size):
        """Resize an image to the target size using high-quality resampling."""
        return image.resize(target_size, Image.Resampling.LANCZOS)

    limit = 10
    limit_count = 0
    for sub_dir in tqdm(os.listdir(base_path), desc="Processing directories"):
        
        if limit_count == limit:
            break
        limit_count += 1

        label = int(sub_dir[1:])
        for sub_dir2 in os.listdir(os.path.join(base_path, sub_dir)):
            # image1 = np.zeros(target_size[::-1], dtype=np.float32)  # Note: np.zeros uses (height, width)
            num_images = 0
            for filename in os.listdir(os.path.join(base_path, sub_dir, sub_dir2)):
                if filename.endswith('.jpg'):
                    file_path = os.path.join(base_path, sub_dir, sub_dir2, filename)
                    img = Image.open(file_path).convert('L')
                    img = resize_image(img, target_size)
                    
                    image1 = np.zeros(target_size[::-1], dtype=np.float32)
                    
                    bw = np.array(img.point(lambda x: 0 if x < 128 else 255), dtype=np.float32)
                    image1 += bw
                    num_images += 1
                    
                    X.append(image1)
                    y.append(label)


    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Save the arrays
    with open('X_10_data.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('y_10_data.pkl', 'wb') as f:
        pickle.dump(y, f)

    # # Load the arrays
    # with open('X_7_data.pkl', 'rb') as f:
    #     X = pickle.load(f)
    # with open('y_7_data.pkl', 'rb') as f:
    #     y = pickle.load(f)

    # print("data loaded")

    # PCA Transformation
    X = X.reshape(X.shape[0], -1)
    pca = PCA(0.99)
    X_pca = pca.fit_transform(X)

    # Splitting data
    x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, shuffle=True)

    # Logistic Regression Training
    lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', verbose=True)
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

    # Save the models to disk
    pickle.dump(lg, open('finalized_model_10_labels.sav', 'wb'))
    pickle.dump(pca, open('pca_model_10_labels.sav', 'wb'))

    return acc


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
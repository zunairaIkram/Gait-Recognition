
# import os
# import numpy as np
# import torch
# import torchvision.transforms as T
# import torchvision
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt

# # Define the DeepLabV3 model with ResNet backbone trained on ADE20K dataset
# deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
# deeplab_model.eval()

# def segment_image_deeplab(image):
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)
    
#     # Define image transformations for preprocessing
#     transform = T.Compose([
#         # Resize the input image to (256, 256)
#         # T.Resize((256, 256)),
#         T.ToTensor(),  # Convert the image to PyTorch tensor
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
#     ])
   
#     input_tensor = transform(image)
#     input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
#     # Perform inference
#     with torch.no_grad():
#         output = deeplab_model(input_batch)['out'][0]
#     output_predictions = output.argmax(0)  # Get the predicted class index for each pixel
    
#     # Convert the index values to a color mask
#     mask = output_predictions.byte().cpu().numpy()
    
#     return mask

# def object_segmentation_api(image, fileName='Seg.png'):
#     # Perform semantic segmentation using DeepLabV3
#     segmentation_mask = segment_image_deeplab(image)
    
#     # Save the segmentation mask
#     cv2.imwrite(fileName, segmentation_mask)
    
#     return segmentation_mask



import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Define the FCN model
fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
fcn_model.eval()

def calculate_image_mean(image):
    # Convert the image to NumPy array
    np_image = np.array(image)

    # Calculate mean values for each channel
    mean_R = np.mean(np_image[:, :, 0])
    mean_G = np.mean(np_image[:, :, 1])
    mean_B = np.mean(np_image[:, :, 2])

    return [mean_R, mean_G, mean_B]

def segment_image_fcn(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Define image transformations for preprocessing
    transform = T.Compose([
        # T.Resize((64, 64)),  
        T.ToTensor(),           # Convert the image to PyTorch tensor
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalize the image
    ])
   
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = fcn_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)  # Get the predicted class index for each pixel
    
    # Convert the index values to a color mask
    mask = output_predictions.byte().cpu().numpy()
    
    return mask

# def refine_silhouette(mask):
#     # Apply morphological closing to smoothen edges and fill small gaps
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     #Dilation, Erosion
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     mask = cv2.erode(mask, kernel, iterations=1)
    
#     return mask

def object_segmentation_api(image, fileName='Seg.png'):
    # Load the input image
    # image = Image.open(image_path) (for pil)
    # Perform semantic segmentation using FCN
    segmentation_mask = segment_image_fcn(image)
    return segmentation_mask
    




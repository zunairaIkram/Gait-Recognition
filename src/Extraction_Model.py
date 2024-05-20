

# # Path to the folder containing video frames
# frames_folder =  r'C:\Users\Fakhi\Desktop\dipProject\train\images\001\scene1_bg_L_090_1'

# # Output folder for saving extracted silhouettes
# output_folder = r'C:\Users\Fakhi\Desktop\dipProject\Gait-Recognition\src\processed'

import cv2
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import time

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
# Segment people only for the purpose of human silhouette extraction
people_class = 15

# Evaluate model
model.eval()
print("Model has been loaded.")

blur = torch.FloatTensor([[[[1.0, 2.0, 1.0],[2.0, 4.0, 2.0],[1.0, 2.0, 1.0]]]]) / 16.0

# Use GPU if supported, for better performance
if torch.cuda.is_available():
    model.to('cuda')
    blur = blur.to('cuda')

# Apply preprocessing (normalization)
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to create segmentation mask
def makeSegMask(img):
    # Scale input frame
    frame_data = torch.FloatTensor(img) / 255.0

    input_tensor = preprocess(frame_data.permute(2, 0, 1))

    # Create mini-batch to be used by the model
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if supported, for better performance
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    segmentation = output.argmax(0)

    bgOut = output[0:1][:][:]
    a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

    people = segmentation.eq(torch.ones_like(segmentation).long().fill_(people_class)).float()

    people.unsqueeze_(0).unsqueeze_(0)

    for i in range(3):
        people = F.conv2d(people, blur, stride=1, padding=1)

    # Activation function to combine masks - F.hardtanh(a * b)
    combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5))))
    combined_mask = combined_mask.expand(1, 3, -1, -1)

    res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

    return res

def Silhoutte_Extraction(img):

        # Apply silhouette extraction to the frame
    mask = makeSegMask(img)

    # Apply thresholding to convert mask to binary map
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Show extracted silhouette only, by multiplying mask and input frame
    final = cv2.bitwise_and(thresh, img)

    return thresh

    #     # # Display the extracted silhouette and mask
    #     # cv2.imshow('Silhouette Mask', mask)
    #     # cv2.imshow('Extracted Silhouette', final)

    #     # Allow early termination with Esc key
    #     key = cv2.waitKey(10)
    #     if key == 27:
    #         break

    # # Release resources
    # cv2.destroyAllWindows()


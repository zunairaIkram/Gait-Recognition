import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import argparse

from model.model import HumanMatting
import inference

def get_silhouette(image, pretrained_weight = r"C:\Users\Fakhi\Desktop\dipProject\Gait-Recognition\src\pretrained\SGHM-ResNet50.pth"):
    # Load Model
    model = HumanMatting(backbone='resnet50')
    device = 'cpu'  # Initialize as CPU, change to 'cuda' if available
    if torch.cuda.is_available():
        device = 'cuda'
        model = nn.DataParallel(model).cuda().eval()
        model.load_state_dict(torch.load(pretrained_weight))
    else:
        state_dict = torch.load(pretrained_weight, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()

    # Convert image to PIL format
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Inference
    pred_alpha, pred_mask = inference.single_inference(model, img, device=device)

    # Convert the silhouette to numpy array
    silhouette = (pred_alpha * 255).astype('uint8')
    
    return silhouette

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Get Silhouette from Image')
#     parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')
#     parser.add_argument('--pretrained-weight', type=str, required=True, help='Path to the pretrained model weight')
#     args = parser.parse_args()

#     # Read the input image
#     input_image = cv2.imread(args.image_path)

#     # Get the silhouette
#     silhouette = get_silhouette(input_image, args.pretrained_weight)

#     # Display the silhouette
#     cv2.imshow('Silhouette', silhouette)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

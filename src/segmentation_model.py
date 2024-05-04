from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
torch.save(model.state_dict(), "./maskRCNN")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold):    
    #img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_labels = pred[0]['labels'].numpy()
    pred_boxes = pred[0]['boxes'].detach().numpy()
    pred_scores = pred[0]['scores'].detach().numpy()

    # Filter predictions for the specified classes
    relevant_indices = [i for i, label in enumerate(pred_labels) 
                        if label in [1, 0] and pred_scores[i] > threshold]

    pred_boxes = pred_boxes[relevant_indices]
    pred_labels = pred_labels[relevant_indices]

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in pred_labels]
    pred_boxes = [[(box[0], box[1]), (box[2], box[3])] for box in pred_boxes]

    return pred_boxes, pred_class
    # pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    # pred_score = list(pred[0]['scores'].detach().numpy())
    # pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    # pred_boxes = pred_boxes[:pred_t+1]
    # pred_class = pred_class[:pred_t+1]
    # return pred_boxes, pred_class

def object_detection_api(img, threshold=0.9, rect_th=3, text_size=3, text_th=3, fileName ='Seg.png'):
    boxes, pred_cls = get_prediction(img, threshold)
    #img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=50)
    print(boxes)
    print(len(boxes))
    for i in range(len(boxes)):
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]),int(boxes[i][1][1]))
        print(rect)
        (mask_grab, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 200, mode = cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask_grab == 2) | (mask_grab == 0), 0, 1).astype('uint8')
        outputMask = (mask2 * 255).astype("uint8")
        img_new = img * mask2[:,:,np.newaxis]
        # plt.figure(figsize=(20,30))
        # plt.imshow((outputMask).astype(np.uint8), cmap = 'gray')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        cv2.imwrite(fileName, outputMask)
        return outputMask

    # plt.figure(figsize=(20,30))
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    

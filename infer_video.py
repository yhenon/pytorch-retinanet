"""
Usage:
python infer_video.py --video_path="sample.avi" --model_path="../coco_resnet_50_map_0_335_state_dict.pt"
"""
import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet.model import resnet50

import glob

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(video_path, model_path):

    # with open(class_list, 'r') as f:
    #     classes = load_classes(csv.reader(f, delimiter=','))
    classes= {0: u'__background__', 
    1: u'person',
    2: u'bicycle',
    3: u'car',
    4: u'motorcycle',
    5: u'airplane',
    6: u'bus',
    7: u'train',
    8: u'truck',
    9: u'boat',
    10: u'traffic light',
    11: u'fire hydrant',
    12: u'stop sign',
    13: u'parking meter',
    14: u'bench',
    15: u'bird',
    16: u'cat',
    17: u'dog',
    18: u'horse',
    19: u'sheep',
    20: u'cow',
    21: u'elephant',
    22: u'bear',
    23: u'zebra',
    24: u'giraffe',
    25: u'backpack',
    26: u'umbrella',
    27: u'handbag',
    28: u'tie',
    29: u'suitcase',
    30: u'frisbee',
    31: u'skis',
    32: u'snowboard',
    33: u'sports ball',
    34: u'kite',
    35: u'baseball bat',
    36: u'baseball glove',
    37: u'skateboard',
    38: u'surfboard',
    39: u'tennis racket',
    40: u'bottle',
    41: u'wine glass',
    42: u'cup',
    43: u'fork',
    44: u'knife',
    45: u'spoon',
    46: u'bowl',
    47: u'banana',
    48: u'apple',
    49: u'sandwich',
    50: u'orange',
    51: u'broccoli',
    52: u'carrot',
    53: u'hot dog',
    54: u'pizza',
    55: u'donut',
    56: u'cake',
    57: u'chair',
    58: u'couch',
    59: u'potted plant',
    60: u'bed',
    61: u'dining table',
    62: u'toilet',
    63: u'tv',
    64: u'laptop',
    65: u'mouse',
    66: u'remote',
    67: u'keyboard',
    68: u'cell phone',
    69: u'microwave',
    70: u'oven',
    71: u'toaster',
    72: u'sink',
    73: u'refrigerator',
    74: u'book',
    75: u'clock',
    76: u'vase',
    77: u'scissors',
    78: u'teddy bear',
    79: u'hair drier',
    80: u'toothbrush'}

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0

    retinanet = resnet50(num_classes=80,)
    retinanet.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model = retinanet  
    

    labels = {}
    for key, value in classes.items():
        labels[key] = value

    
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    rows, cols, cns = image.shape
    size = (cols,rows)
    
    out = cv2.VideoWriter('output3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    while success:
        success,image = vidcap.read()
        
        if (not success) or (image is None):
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                #print(int(classification[idxs[0][j]]))
                label_name = str(int(classification[idxs[0][j]]))
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            
            out.write(image_orig)
    out.release()  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--video_path', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    

    parser = parser.parse_args()

    


    

    detect_image(parser.video_path, parser.model_path)

    
    
    


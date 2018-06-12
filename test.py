import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pdb
import time
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

assert torch.__version__.split('.')[1] == '4'

import sys
import cv2

print('CUDA available: {}'.format(torch.cuda.is_available()))

coco = False

if coco:
	dataset_train = CocoDataset('../coco/', set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
	dataset_val = CocoDataset('../coco/', set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
else:
	dataset_train = CSVDataset(train_file='/home/gmautomap/../bcaine/data/MightAI_CSV/train_labels.csv', class_list='/home/gmautomap/../bcaine/data/MightAI_CSV/class_idx.csv', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
	dataset_val = CSVDataset(train_file='/home/gmautomap/../bcaine/data/MightAI_CSV/train_labels.csv', class_list='/home/gmautomap/../bcaine/data/MightAI_CSV/class_idx.csv', transform=transforms.Compose([Normalizer(), Resizer()]))


sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

model = torch.load('csv_model_3.pt')

use_gpu = True

if use_gpu:
	model = model.cuda()

model.eval()

unnormalize = UnNormalizer()


def draw_caption(image, box, caption):

	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

for idx, data in enumerate(dataloader_val):

	scores, classification, transformed_anchors = model(data['img'].cuda().float())

	idxs = np.where(scores>0.5)
	img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

	img[img<0] = 0
	img[img>255] = 255

	img = np.transpose(img, (1, 2, 0))

	img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

	for j in range(idxs[0].shape[0]):
		bbox = transformed_anchors[idxs[0][j], :]
		x1 = int(bbox[0])
		y1 = int(bbox[1])
		x2 = int(bbox[2])
		y2 = int(bbox[3])
		label_name = dataset_val.labels[int(classification[idxs[0][j]])]
		draw_caption(img, (x1, y1, x2, y2), label_name)

		cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
		print(label_name)

	cv2.imshow('img', img)
	cv2.waitKey(0)

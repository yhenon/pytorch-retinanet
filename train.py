import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import model
from anchors import Anchors
import losses
import pdb
import time
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

assert torch.__version__.split('.')[1] == '4'
import requests
import coco_eval
import collections
import sys

print('CUDA available: {}'.format(torch.cuda.is_available()))

coco = False

if coco:
	dataset_train = CocoDataset('../coco/', set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
	dataset_val = CocoDataset('../coco/', set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
else:
	dataset_train = CSVDataset(train_file='/home/gmautomap/../bcaine/data/MightAI_CSV/train_labels.csv', class_list='/home/gmautomap/../bcaine/data/MightAI_CSV/class_idx.csv', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
	dataset_val = CSVDataset(train_file='/home/gmautomap/../bcaine/data/MightAI_CSV/train_labels.csv', class_list='/home/gmautomap/../bcaine/data/MightAI_CSV/class_idx.csv', transform=transforms.Compose([Normalizer(), Resizer()]))

model = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)

sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

use_gpu = True

if use_gpu:
	model = model.cuda()

model.training = True

#model_old = torch.load('sgd_model_14.pt')

#model_state = model_old.state_dict()

#model.load_state_dict(model_state)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

total_loss = losses.loss

loss_hist = collections.deque(maxlen=500)

model.train()
model.freeze_bn()

num_epochs = 100

print('Num training images: {}'.format(len(dataset_train)))

for epoch_num in range(num_epochs):

	model.train()
	model.freeze_bn()
	
	epoch_loss = []
	
	for iter_num, data in enumerate(dataloader_train):
		try:
			
			optimizer.zero_grad()

			classification, regression, anchors = model(data['img'].cuda().float())
			
			classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

			loss = classification_loss + regression_loss
			
			if bool(loss == 0):
				continue

			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

			optimizer.step()

			loss_hist.append(float(loss))

			epoch_loss.append(float(loss))

			print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

		except Exception as e:
			print(e)
			pdb.set_trace()
	
	print('Evaluating dataset')
	
	if coco:
		coco_eval.evaluate_coco(dataset_val, model)
	
	scheduler.step(np.mean(epoch_loss))	
	torch.save(model, 'csv_model_{}.pt'.format(epoch_num))

model.eval()

torch.save(model, 'model_final.pt'.format(epoch_num))

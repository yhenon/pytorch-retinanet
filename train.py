import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

	parser = parser.parse_args(args)

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on COCO,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')


		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	total_loss = losses.loss

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):

		retinanet.train()
		retinanet.freeze_bn()
		
		epoch_loss = []
		
		for iter_num, data in enumerate(dataloader_train):
			try:
				optimizer.zero_grad()

				classification, regression, anchors = retinanet(data['img'].cuda().float())
				
				classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

				loss = classification_loss + regression_loss
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

			except Exception as e:
				print(e)
		
		
		if parser.dataset == 'coco':
			print('Evaluating dataset')

			coco_eval.evaluate_coco(dataset_val, retinanet)

		elif parser.dataset == 'csv' and parser.csv_val is not None:
			print('Evaluating dataset')

			total_loss_joint = 0.0
			total_loss_classification = 0.0
			total_loss_regression = 0.0

			for iter_num, data in enumerate(dataloader_val):

				if iter_num % 100 == 0:
					print('{}/{}'.format(iter_num, len(dataset_val)))

				with torch.no_grad():			
					classification, regression, anchors = retinanet(data['img'].cuda().float())
					
					classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

					total_loss_joint += float(classification_loss + regression_loss)
					total_loss_regression += float(regression_loss)
					total_loss_classification += float(classification_loss)

			total_loss_joint /= float(len(dataset_val))
			total_loss_classification /= float(len(dataset_val))
			total_loss_regression /= float(len(dataset_val))

			print('Validation epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Total loss: {:1.5f}'.format(epoch_num, float(total_loss_classification), float(total_loss_regression), float(total_loss_joint)))
		
		
		scheduler.step(np.mean(epoch_loss))	

		torch.save(retinanet, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

	retinanet.eval()

	torch.save(retinanet, 'model_final.pt'.format(epoch_num))

if __name__ == '__main__':
 main()
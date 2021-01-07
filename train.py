import argparse
import collections
import configparser
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorchtools import EarlyStopping
from retinanet import csv_eval
from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--configfile', help='Path to the config file', default='config.txt', type=str)

    parser = parser.parse_args(args)

    configs = configparser.ConfigParser()
    configs.read(parser.configfile)

    try:
        batchsize = int(configs['TRAINING']['batchsize'])
        depth = int(configs['TRAINING']['depth'])
        maxepochs = int(configs['TRAINING']['maxepochs'])
        maxside = int(configs['TRAINING']['maxside'])
        minside = int(configs['TRAINING']['minside'])
        savepath = configs['TRAINING']['savepath']
        try:
            ratios = json.loads(configs['MODEL']['ratios'])
            scales = json.loads(configs['MODEL']['scales'])
        except Exception as e:
            print(e)
            print('USING DEFAULT RATIOS AND SCALES')
            ratios = None
            scales = None
    except Exception as e:
        print(e)
        print('CONFIG FILE IS INVALID. PLEASE REFER TO THE EXAMPLE CONFIG FILE AT config.txt')
        sys.exit()

    model_save_dir = datetime.now().strftime("%d_%b_%Y_%H_%M") if savepath == 'datetime' else savepath

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)

    # Copy the config file into the model save directory
    shutil.copy(parser.configfile, os.path.join(model_save_dir, 'config.txt'))
    # Create the data loaders
    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train,')

    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes,')

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(),
                                                             Resizer(min_side=minside,
                                                                     max_side=maxside)]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer(min_side=minside,
                                                                                     max_side=maxside)]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batchsize, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    dataloader_val = None

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, ratios=ratios,
                                   scales=scales)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, ratios=ratios,
                                   scales=scales)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, ratios=ratios,
                                   scales=scales)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, ratios=ratios,
                                    scales=scales)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, ratios=ratios,
                                    scales=scales)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.25, cooldown=1,
                                                     min_lr=1e-9)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    earlystopping = EarlyStopping(patience=10, verbose=True, delta=1e-10,
                                  path=os.path.join(model_save_dir, 'best_model.pt'))
    print('Num training images: {}'.format(len(dataset_train)))

    loss_dict = OrderedDict()
    val_loss_dict = OrderedDict()

    for epoch_num in range(maxepochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_val_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)),
                    end='\r', flush=True)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if (len(epoch_loss)):
            loss_dict[epoch_num] = np.mean(epoch_loss)

        print('')

        if dataloader_val is not None:
            print('Evaluating dataset')
            for iter_num, data in enumerate(dataloader_val):
                try:
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            val_classification_loss, val_regression_loss = retinanet(
                                [data['img'].cuda().float(), data['annot']])
                        else:
                            val_classification_loss, val_regression_loss = retinanet(
                                [data['img'].float(), data['annot']])

                        val_classification_loss = val_classification_loss.mean()
                        val_regression_loss = val_classification_loss.mean()

                        val_loss = val_classification_loss + val_regression_loss
                        print('Validation Loss: {:1.5f}'.format(val_loss), end='\r', flush=True)
                        epoch_val_loss.append(float(val_loss))

                except Exception as e:
                    print(e)
                    continue
            print('')
            if (len(epoch_val_loss)):
                val_loss_dict[epoch_num] = np.mean(epoch_val_loss)

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            print('-----------------')
            print(mAP)
            print('-----------------')
        scheduler.step(np.mean(epoch_loss))

        model_save_path = os.path.join(model_save_dir, f'retinanet_{epoch_num}.pt')
        torch.save(retinanet.module, model_save_path)
        print(f'Saved model of epoch {epoch_num} to {model_save_path}')

        earlystopping(val_loss_dict[epoch_num], retinanet)

        if earlystopping.early_stop:
            print("Early stopping")
            break

    retinanet.eval()

    torch.save(retinanet, os.path.join(model_save_dir, 'model_final.pt'))

    with open(os.path.join(model_save_dir, 'loss_history.txt'), 'w') as f:
        for epoch_num, loss in loss_dict.items():
            f.write(f'{epoch_num}:{loss} \n')
    with open(os.path.join(model_save_dir, 'val_loss_history.txt'), 'w') as f:
        for epoch_num, loss in val_loss_dict.items():
            f.write(f'{epoch_num}:{loss} \n')


if __name__ == '__main__':
    main()

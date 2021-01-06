import argparse
import collections
import json
import os
from collections import OrderedDict

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
    parser.add_argument('--batchsize', help='batch size ', type=int, default=1)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--savepath', help='Where to save the model', type=str, default='models')

    parser = parser.parse_args(args)

    if not os.path.exists(parser.savepath):
        os.makedirs(parser.savepath, exist_ok=True)

    # Create the data loaders
    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train,')

    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes,')

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batchsize, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    dataloader_val = None

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

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
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    earlystopping = EarlyStopping(patience=10, verbose=True, delta=1e-10,
                                  path=os.path.join(parser.savepath, 'best_model.pt'))
    print('Num training images: {}'.format(len(dataset_train)))

    loss_dict = OrderedDict()
    val_loss_dict = OrderedDict()

    for epoch_num in range(parser.epochs):

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
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        val_classification_loss, val_regression_loss = retinanet(
                            [data['img'].cuda().float(), data['annot']])
                    else:
                        val_classification_loss, val_regression_loss = retinanet([data['img'].float(), data['annot']])

                    val_classification_loss = val_classification_loss.mean()
                    val_regression_loss = val_classification_loss.mean()

                    val_loss = val_classification_loss + val_regression_loss
                    print('Validation Loss: {:1.5f}'.format(val_loss), end='\r', flush=True)
                    epoch_val_loss.append(val_loss.numpy())

                except Exception as e:
                    print(e)
                    continue
                print('')
            if (len(epoch_val_loss)):
                val_loss_dict[epoch_num] = np.mean(epoch_val_loss)

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            print(mAP)
        scheduler.step(np.mean(epoch_loss))

        model_save_path = os.path.join(parser.savepath, f'retinanet_{epoch_num}.pt')
        torch.save(retinanet.module, model_save_path)
        print(f'Saved model of epoch {epoch_num} to {model_save_path}')

        earlystopping(list(val_loss_dict.values()), retinanet)

        if earlystopping.early_stop:
            print("Early stopping")
            break

    retinanet.eval()

    torch.save(retinanet, os.path.join(parser.savepath, 'model_final.pt'))
    with open(os.path.join(parser.savepath, 'loss_history.json'), 'w') as f:
        json.dump(loss_dict, f)
    with open(os.path.join(parser.savepath, 'val_loss_history.json'), 'w') as f:
        json.dump(val_loss_dict, f)


if __name__ == '__main__':
    main()

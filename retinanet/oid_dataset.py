from __future__ import print_function, division

import csv
import json
import os
import warnings

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
from PIL import Image
from torch.utils.data import Dataset


def get_labels(metadata_dir, version='v4'):
    if version == 'v4' or version == 'challenge2018':
        csv_file = 'class-descriptions-boxable.csv' if version == 'v4' else 'challenge-2018-class-descriptions-500.csv'

        boxable_classes_descriptions = os.path.join(metadata_dir, csv_file)
        id_to_labels = {}
        cls_index = {}

        i = 0
        with open(boxable_classes_descriptions) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    label = row[0]
                    description = row[1].replace("\"", "").replace("'", "").replace('`', '')

                    id_to_labels[i] = description
                    cls_index[label] = i

                    i += 1
    else:
        trainable_classes_path = os.path.join(metadata_dir, 'classes-bbox-trainable.txt')
        description_path = os.path.join(metadata_dir, 'class-descriptions.csv')

        description_table = {}
        with open(description_path) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    description_table[row[0]] = row[1].replace("\"", "").replace("'", "").replace('`', '')

        with open(trainable_classes_path, 'rb') as f:
            trainable_classes = f.read().split('\n')

        id_to_labels = dict([(i, description_table[c]) for i, c in enumerate(trainable_classes)])
        cls_index = dict([(c, i) for i, c in enumerate(trainable_classes)])

    return id_to_labels, cls_index


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version='v4'):
    validation_image_ids = {}

    if version == 'v4':
        annotations_path = os.path.join(metadata_dir, subset, '{}-annotations-bbox.csv'.format(subset))
    elif version == 'challenge2018':
        validation_image_ids_path = os.path.join(metadata_dir, 'challenge-2018-image-ids-valset-od.csv')

        with open(validation_image_ids_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=['ImageID'])
            reader.next()
            for line, row in enumerate(reader):
                image_id = row['ImageID']
                validation_image_ids[image_id] = True

        annotations_path = os.path.join(metadata_dir, 'challenge-2018-train-annotations-bbox.csv')
    else:
        annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    fieldnames = ['ImageID', 'Source', 'LabelName', 'Confidence',
                  'XMin', 'XMax', 'YMin', 'YMax',
                  'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']

    id_annotations = dict()
    with open(annotations_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        next(reader)

        images_sizes = {}
        for line, row in enumerate(reader):
            frame = row['ImageID']

            if version == 'challenge2018':
                if subset == 'train':
                    if frame in validation_image_ids:
                        continue
                elif subset == 'validation':
                    if frame not in validation_image_ids:
                        continue
                else:
                    raise NotImplementedError('This generator handles only the train and validation subsets')

            class_name = row['LabelName']

            if class_name not in cls_index:
                continue

            cls_id = cls_index[class_name]

            if version == 'challenge2018':
                # We recommend participants to use the provided subset of the training set as a validation set.
                # This is preferable over using the V4 val/test sets, as the training set is more densely annotated.
                img_path = os.path.join(main_dir, 'images', 'train', frame + '.jpg')
            else:
                img_path = os.path.join(main_dir, 'images', subset, frame + '.jpg')

            if frame in images_sizes:
                width, height = images_sizes[frame]
            else:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.width, img.height
                        images_sizes[frame] = (width, height)
                except Exception as ex:
                    if version == 'challenge2018':
                        raise ex
                    continue

            x1 = float(row['XMin'])
            x2 = float(row['XMax'])
            y1 = float(row['YMin'])
            y2 = float(row['YMax'])

            x1_int = int(round(x1 * width))
            x2_int = int(round(x2 * width))
            y1_int = int(round(y1 * height))
            y2_int = int(round(y2 * height))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if y2_int == y1_int:
                warnings.warn('filtering line {}: rounding y2 ({}) and y1 ({}) makes them equal'.format(line, y2, y1))
                continue

            if x2_int == x1_int:
                warnings.warn('filtering line {}: rounding x2 ({}) and x1 ({}) makes them equal'.format(line, x2, x1))
                continue

            img_id = row['ImageID']
            annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

            if img_id in id_annotations:
                annotations = id_annotations[img_id]
                annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}
    return id_annotations


class OidDataset(Dataset):
    """Oid dataset."""

    def __init__(self, main_dir, subset, version='v4', annotation_cache_dir='.', transform=None):
        if version == 'v4':
            metadata = '2018_04'
        elif version == 'challenge2018':
            metadata = 'challenge2018'
        elif version == 'v3':
            metadata = '2017_11'
        else:
            raise NotImplementedError('There is currently no implementation for versions older than v3')

        self.transform = transform

        if version == 'challenge2018':
            self.base_dir = os.path.join(main_dir, 'images', 'train')
        else:
            self.base_dir = os.path.join(main_dir, 'images', subset)

        metadata_dir = os.path.join(main_dir, metadata)
        annotation_cache_json = os.path.join(annotation_cache_dir, subset + '.json')

        self.id_to_labels, cls_index = get_labels(metadata_dir, version=version)

        if os.path.exists(annotation_cache_json):
            with open(annotation_cache_json, 'r') as f:
                self.annotations = json.loads(f.read())
        else:
            self.annotations = generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index,
                                                                version=version)
            json.dump(self.annotations, open(annotation_cache_json, "w"))

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

        # (label -> name)
        self.labels = self.id_to_labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def image_path(self, image_index):
        path = os.path.join(self.base_dir, self.id_to_image_id[image_index] + '.jpg')
        return path

    def load_image(self, image_index):
        path = self.image_path(image_index)
        img = skimage.io.imread(path)

        if len(img.shape) == 1:
            img = img[0]

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        try:
            return img.astype(np.float32) / 255.0
        except Exception:
            print (path)
            exit(0)

    def load_annotations(self, image_index):
        # get ground truth annotations
        image_annotations = self.annotations[self.id_to_image_id[image_index]]

        labels = image_annotations['boxes']
        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes

    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id_to_image_id[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    def num_classes(self):
        return len(self.id_to_labels)

# pytorch-retinanet

![img3](https://github.com/yhenon/pytorch-retinanet/blob/master/images/3.jpg)
![img5](https://github.com/yhenon/pytorch-retinanet/blob/master/images/5.jpg)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.


## Results
Currently, this repo achieves 33.7% mAP at 600px resolution with a Resnet-50 backbone. The published result is 34.0% mAP. The difference is likely due to the use of Adam optimizer instead of SGD with weight decay.

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install cffi

pip install pandas

pip install pycocotools

pip install cython

pip install pycocotools

pip install opencv-python

pip install requests

```

4) Build the NMS extension.

```
cd pytorch-retinanet/lib
bash make.sh
cd ../
```

## Training

The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

```
python train.py coco <path/to/coco>
```

For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py csv <path/to/annotations.csv> <path/to/classes.csv>
```

## Visualization

To visualize the network detection, use `test.py`.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Acknowledgements

- Significant amounts of code are borrowed from the [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)

## Examples

![img1](https://github.com/yhenon/pytorch-retinanet/blob/master/images/1.jpg)
![img2](https://github.com/yhenon/pytorch-retinanet/blob/master/images/2.jpg)
![img4](https://github.com/yhenon/pytorch-retinanet/blob/master/images/4.jpg)
![img6](https://github.com/yhenon/pytorch-retinanet/blob/master/images/6.jpg)
![img7](https://github.com/yhenon/pytorch-retinanet/blob/master/images/7.jpg)
![img8](https://github.com/yhenon/pytorch-retinanet/blob/master/images/8.jpg)

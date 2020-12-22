import argparse
import collections
import datetime
import os

import neptune
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from dbt_comp.data.duke import DukeDatasetV1, Resizer, collater, RepeatChannels, NumPyToTorch, Standardizer, SimpleAugmenter
# from retinanet import coco_eval
# from retinanet import csv_eval
from retinanet import model
# from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
#     Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser('Simple training script for training a RetinaNet network.')
    parser.add_argument('-p', '--project', type=str, default='duke_data_v2_loader_v1.1', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--lms', type=boolean_string, default=False,
                        help='use large model support on skynet')
    parser.add_argument('--neptune_exp_name', type=str, default='duke-training', help='Name of the experiment in Neptune.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco or duke.', default='duke')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--loader-version', help='Version of Duke dataset', type=int, default=1)

    parsed_args = parser.parse_args()
    return parsed_args


def init_neptune(opt, params):
    neptune.init(f'bteam/{opt.neptune_exp_name}',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNTQ1NWMxMDYtMTI1NC00NTFjLWExMzktMWJhYzc0ZDUxMTcyIn0=")
    experiment_params = {}
    for arg in vars(opt):
        print(arg, getattr(opt, arg))
        experiment_params[arg] = getattr(opt, arg)

    # additional params from yml file
    for k, v in params.params.items():
        print(k, v)
        experiment_params[k] = v

    # Create experiment with defined parameters
    neptune.create_experiment(name='Training of a DBT model',
                              params=experiment_params)


def train(opt):

    params = Params(f'projects/{opt.project}.yml')

    if params.params is None:
        raise KeyError(f'projects/{opt.project}.yml is incorrect or deprecated. Please use newest configuration file.')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    train_loader_pipeline = [
        Resizer(input_sizes[opt.compound_coef]),
        Standardizer(),  # standardize input instead of normalizing
        NumPyToTorch(),
        RepeatChannels(3),
    ]

    filter_negatives_training = params.params.get('filter_negatives_training', True)
    augmentation_version = params.params.get('augmentation_version', None)

    if augmentation_version is None:
        pass  # this is fine
    elif (augmentation_version.lower() == 'simple'):
        train_pipeline = [SimpleAugmenter()] + train_loader_pipeline
    elif augmentation_version is not None:
        raise KeyError(augmentation_version)

    train_transform = transforms.Compose(train_loader_pipeline)

    val_transform = transforms.Compose(
        [
            Resizer(input_sizes[opt.compound_coef]),
            Standardizer(),  # standardize input instead of normalizing
            NumPyToTorch(),
            RepeatChannels(3),
        ]
    )

    # # Create the data loaders
    # if opt.dataset == 'csv':
    #
    #     if opt.csv_train is None:
    #         raise ValueError('Must provide --csv_train when training on COCO,')
    #
    #     if opt.csv_classes is None:
    #         raise ValueError('Must provide --csv_classes when training on COCO,')
    #
    #     dataset_train = CSVDataset(train_file=opt.csv_train, class_list=opt.csv_classes,
    #                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    #
    #     if opt.csv_val is None:
    #         dataset_val = None
    #         print('No validation annotations provided.')
    #     else:
    #         dataset_val = CSVDataset(train_file=opt.csv_val, class_list=opt.csv_classes,
    #                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    if opt.dataset == 'duke':

        if opt.loader_version == 1:
            selected_dataset = DukeDatasetV1
        else:
            raise KeyError(opt.loader_version)

        dataset_train = selected_dataset(root_dir=params.data_path, set=params.train_set,
                                        slice_neighbor_sampling=params.train_slice_neighbor,
                                        filter_negatives=filter_negatives_training,
                                        random_patch_selection=True,
                                        obj_list=params.obj_list,
                                        transform=train_transform)
        dataloader_train = DataLoader(dataset_train, **training_params)

        dataset_val = selected_dataset(root_dir=params.data_path, set=params.val_set,
                                   slice_neighbor_sampling=params.val_slice_neighbor,
                                   filter_negatives=True,
                                   random_patch_selection=False,
                                   obj_list=params.obj_list,
                                   transform=val_transform)
        dataloader_val = DataLoader(dataset_val, **val_params)

        # dataset_train = CocoDataset(opt.coco_path, set_name='train2017',
        #                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # dataset_val = CocoDataset(opt.coco_path, set_name='val2017',
        #                           transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv, coco or duke), exiting.')
    #
    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)  # TODO: should I keep this / use these params? check.
    # dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    #
    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if opt.depth == 18:
        retinanet = model.resnet18(num_classes=len(params.obj_list), pretrained=True)
    elif opt.depth == 34:
        retinanet = model.resnet34(num_classes=len(params.obj_list), pretrained=True)
    elif opt.depth == 50:
        retinanet = model.resnet50(num_classes=len(params.obj_list), pretrained=True)
    elif opt.depth == 101:
        retinanet = model.resnet101(num_classes=len(params.obj_list), pretrained=True)
    elif opt.depth == 152:
        retinanet = model.resnet152(num_classes=len(params.obj_list), pretrained=True)
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

    print('Num training images: {}'.format(len(dataset_train)))

    # initialize neptune logging
    init_neptune(opt, params)
    # decide saved_path using neptune id
    opt.saved_path = opt.saved_path + f'/{params.project_name}/{neptune.get_experiment().id}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(opt.saved_path, exist_ok=True)
    os.system("cp " + f'projects/{opt.project}.yml' + " " + os.path.join(opt.saved_path, f'{opt.project}.yml'))

    for epoch_num in range(opt.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                neptune.log_metric('Loss', loss)
                neptune.log_metric('Regression_loss', regression_loss)
                neptune.log_metric('Classification_loss', classification_loss)

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # if opt.dataset == 'coco':
        #
        #     print('Evaluating dataset')
        #
        #     coco_eval.evaluate_coco(dataset_val, retinanet)
        #
        # elif opt.dataset == 'csv' and opt.csv_val is not None:
        #
        #     print('Evaluating dataset')
        #
        #     mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(opt.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    args = get_args()
    train(args)

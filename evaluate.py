import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
import networks
from dataset.datasets import CSDataSet
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.pyt_utils import load_model

from engine import Engine
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
BATCH_SIZE = 4
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
INPUT_SIZE = '769,769'
RESTORE_FROM = './deeplab_resnet.ckpt'

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of recurrence.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    return parser

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[0], image_size[2], image_size[3], classes))
    count_predictions = np.zeros((1, image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # print("Predicting tile %i" % tile_counter)
            padded_prediction = net(torch.from_numpy(padded_img).cuda(non_blocking=True))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().numpy().transpose(0,2,3,1)
            prediction = padded_prediction[0, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[0, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    N_, C_, H_, W_ = image.shape
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    prediction = net(image.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().numpy().transpose(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        # scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        scaled_probs = predict_sliding(net, scale_image, tile_size, classes, recurrence)
        if flip_evaluation == True:
            # flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
            flip_scaled_probs = predict_sliding(net, scale_image[:,:,:,::-1].copy(), tile_size, classes, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True

        h, w = map(int, args.input_size.split(','))
        if args.whole:
            input_size = (1024, 2048)
        else:
            input_size = (h, w)

        seg_model = eval('networks.' + args.model + '.Seg_Model')(
            num_classes=args.num_classes
        )
        
        load_model(seg_model, args.restore_from)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        model.eval()

        dataset = CSDataSet(args.data_dir, args.data_list, crop_size=(1024, 2048), mean=IMG_MEAN, scale=False, mirror=False)
        test_loader, test_sampler = engine.get_test_loader(dataset)

        if engine.distributed:
            test_sampler.set_epoch(0)

        data_list = []
        confusion_matrix = np.zeros((args.num_classes,args.num_classes))
        palette = get_palette(256)

        save_path = os.path.join(os.path.dirname(args.restore_from), 'outputs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(test_loader)

        for idx in pbar:
            image, label, size, name = dataloader.next()
            size = size[0].numpy()
            with torch.no_grad():
                output = predict_multiscale(model, image, input_size, [1.0], args.num_classes, False, 0)


            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            seg_gt = np.asarray(label.numpy()[:,:size[0],:size[1]], dtype=np.int)

            for i in range(image.size(0)): 
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(os.path.join(save_path, name[i]+'.png'))
        
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            # show_all(gt, output)
            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

            print_str = ' Iter{}/{}'.format(idx + 1, len(test_loader))
            pbar.set_description(print_str, refresh=False)

        confusion_matrix = torch.from_numpy(confusion_matrix).contiguous().cuda()
        confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        
        # getConfusionMatrixPlot(confusion_matrix)
        if engine.distributed and engine.local_rank == 0:
            print({'meanIU':mean_IU, 'IU_array':IU_array})
            model_path = os.path.dirname(args.restore_from)
            with open(os.path.join(model_path, 'result.txt'), 'w') as f:
                f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist()}))

if __name__ == '__main__':
    main()

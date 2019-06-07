import argparse

from keras.models import load_model
import glob
import cv2
import os
import numpy as np
import random
from tqdm import tqdm
from .train import find_latest_checkpoint 
from .data_utils.data_loader import get_image_arr, get_segmentation_arr
import json
from .models.config import IMAGE_ORDERING
from . import metrics
from .models import model_from_name

import six

random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]


def model_from_checkpoint_path(checkpoints_path):
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (not latest_weights is None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'],
                                                         input_height=model_config['input_height'],
                                                         input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    print("seg_img before resize", seg_img.shape)
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (not inp_dir is None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(os.path.join(inp_dir, "*.png")) + glob.glob(
            os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        all_prs.append(pr)

    return all_prs


def get_pr(model, inp):
    orininal_h = 192 
    orininal_w = 320

    seg_img = predict(model, inp)

    unique_elements, counts_elements = np.unique(seg_img, return_counts=True)
    print("Frequency of unique values of pr array:")
    print(np.asarray((unique_elements, counts_elements)))

    seg_img = np.stack((seg_img.astype('uint8'), seg_img.astype('uint8'), seg_img.astype('uint8')))
    seg_img = np.reshape(seg_img, (96, 160, 3)) 
    print('stack_seg_img size', seg_img.shape)
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    return seg_img 

def evaluate(annots_dir=None, images_dir=None, checkpoints_path=None):
    model = model_from_checkpoint_path(checkpoints_path)

    images = None
    annots = None
    
    if images_dir is not None:
        images = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(
            os.path.join(images_dir, "*.jpeg"))
    
    if annots_dir is not None:
        annots = glob.glob(os.path.join(annots_dir, "*.jpg")) + glob.glob(os.path.join(annots_dir, "*.png")) + glob.glob(
            os.path.join(annots_dir, "*.jpeg"))
    
    ious = []
    precisions = []
    recalls = []
    for inp, ann in tqdm(zip(images, annots)):
        gt = get_segmentation_arr(ann, 2, 320, 192)
        print('gt shape', gt.shape)
        gt = gt.argmax(-1)
        pr = get_segmentation_arr(get_pr(model, inp), 2, 320, 192)
        pr = pr.argmax(-1)
        print('pr shape', pr.shape)

        unique_elements, counts_elements = np.unique(gt, return_counts=True)
        print("Frequency of unique values of gt array:")
        print(np.asarray((unique_elements, counts_elements)))

        unique_elements, counts_elements = np.unique(pr, return_counts=True)
        print("Frequency of unique values of pr array after resize:")
        print(np.asarray((unique_elements, counts_elements)))
        precision = metrics.get_precision(gt, pr, 2)
        iou = metrics.get_iou(gt, pr, 2)
        recall = metrics.get_recall(gt, pr, 2)

        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
    ious = np.array(ious)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    print("Class wise IoU ", np.mean(ious, axis=0))
    print("Total  IoU ", np.mean(ious))
    print("Class wise precision ", np.mean(precisions, axis=0))
    print("Total precision ", np.mean(precisions))
    print("Class wise recall ", np.mean(recalls, axis=0))
    print("Total recall", np.mean(recalls))
    


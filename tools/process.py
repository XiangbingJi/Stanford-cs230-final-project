import argparse
import cv2
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# This file takes in the TuSimple json labels and convert it to a binary image. The binary image is the same size with
# the original image and be consumed by our semantic segmentation model.
def process():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--home_dir", required=True,
                    help="path to the HOME_DIR")
    args = vars(ap.parse_args())

    HOME_DIR = args["home_dir"] + '/'

    json_gt = [json.loads(line) for line in open(HOME_DIR + 'label_data.json')]

    for i, gt in enumerate(json_gt):

        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']

        img = cv2.imread(HOME_DIR + raw_file)

        gt_lanes_label = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        img_label = np.zeros(img.shape)

        for lane in gt_lanes_label:
            cv2.polylines(img_label, np.int32([lane]), isClosed=False, color=(1, 1, 1), thickness=5)

        generated_file = "_".join(raw_file.split('/')[:-1]) + ".png"

        plt.imsave(HOME_DIR + 'original_image/' + generated_file, img)
        plt.imsave(HOME_DIR + 'label_image/' + generated_file, img_label)

        if i % 10 == 0:
            print (i)
if __name__ == "__main__":
    process()

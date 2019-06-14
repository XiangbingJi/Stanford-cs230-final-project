# Stanford-cs230-final-project

This project uses a semantic segmentation approach to tackle the freeway lane detection problem. 

## Dataset
We use the TuSimple dataset: https://github.com/TuSimple/tusimple-benchmark, which contains 2858 images. And we split the data into train (80%), dev (10%), test (10%).

### pre-processing
use ./tools/process to pre-process our data. The script takes in the label of TuSimple Dataset (in Json format) and generates a binary image for the semantic segmentation. Usage as follows:

```
python tools/process.py --home_dir
```

The home_dir is where you store all the clips of your dataset. The script saves its outputs in two separate folders. home_dir/original_image saves all original images, where home_dir/label_image saves all binary label images. The image names in original_image folder and label_image folder are identical. For example, an image in path home_dir/clips/0313-1/60/20.jpg will generate two images named clips_0313-1_60.png in both home_dir/original_image and home_dir/label_image.

### post-processing

To start post-processing, you need to be read three text files: train.txt, val.txt, test.txt, where each one of them contain file names for train, dev and test set respectively. 
  

The video demo on a video that is outside of our training/dev/testing data set. 

![](demo.gif)

The baseline model contains the code cloned from an existing implementation
(https://github.com/tkuanlun350/Tensorflow-SegNet) for
SegNet (https://arxiv.org/pdf/1511.00561v3.pdf). 

We will start from the baseline mode and build our lane detection model. 

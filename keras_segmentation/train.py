import keras_metrics
import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset
from .models import model_from_name
import tensorflow as tf
import os
import six
import pickle


def find_latest_checkpoint(checkpoints_path):
    ep = 0
    r = None
    while True:
        if os.path.isfile(checkpoints_path + "." + str(ep)):
            r = checkpoints_path + "." + str(ep)
        else:
            return r

        ep += 1


def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    class_weights = [1., 500.]  # set your class weights here
    # computer weights based on onehot labels
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss


def my_recall(actual, predicted):
    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    return tf.divide(TP, TP + FN)

def my_precision(actual, predicted):
    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    return tf.divide(TP, TP + FP)

def my_f1(actual, predicted):
    recall = my_recall(actual, predicted)
    precision = my_precision(actual, predicted)
    return tf.divide(2 * precision * recall, precision + recall)


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=30,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=10,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=1,
          optimizer_name='adadelta'
          ):
    if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
        # create the model from the name
        assert (not n_classes is None), "Please provide the n_classes"
        if (not input_height is None) and (not input_width is None):
            model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert not (val_images is None)
        assert not (val_annotations is None)

    if not optimizer_name is None:
        model.compile(loss=my_weighted_loss,
                      optimizer=optimizer_name,
                      metrics=[keras_metrics.precision(), keras_metrics.recall(), my_recall, my_precision, my_f1])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "model_class": model.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    verify_dataset = False
    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, n_classes, input_height,
                                             input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                               input_width, output_height, output_width)

    print(model.summary())
    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=12,
                                          epochs=1)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)
            print("Saving history", history.history)
            with open('./history.json', 'ab') as f:
                pickle.dump(history.history, f)

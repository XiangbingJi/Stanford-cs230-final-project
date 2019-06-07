import numpy as np

EPS = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum(np.logical_and((gt == cl), (pr == cl)))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = intersection / (union + EPS)
        class_wise[cl] = iou
    return class_wise


def get_recall(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)    
    for cl in range(n_classes):
        true_positives = np.sum(np.logical_and((gt == pr), (pr == cl)))
        possible_positives = np.sum((gt == cl))
        recall = true_positives / (possible_positives + EPS)
        class_wise[cl] = recall
    return class_wise


def get_precision(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        true_positives = np.sum(np.logical_and((gt == pr), (pr == cl)))
        predicted_positives = np.sum((pr == cl))
        precision = true_positives / (predicted_positives + EPS)
        class_wise[cl] = precision
    return class_wise

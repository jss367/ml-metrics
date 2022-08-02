import sys

EPSILON = sys.float_info.epsilon

def calc_precision(tp, fp, epsilon=EPSILON):
    """
    Returns 1 in edge case where fp=0 because there were no incorrect predictions
    """
    return (tp + epsilon) / (tp + fp + epsilon)


def calc_recall(tp, fn, epsilon=EPSILON):
    """
    Returns 1 in edge case where fn=0 because 100% of tp were discovered
    """
    return (tp + epsilon) / (tp + fn + epsilon)


def calc_accuracy(tp, tn, fp, fn, epsilon=EPSILON):
    """
    Returns 1 in edge case where all values are 0
    """
    return (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)


def calc_misclassification(tp, tn, fp, fn, epsilon=EPSILON):
    """
    Returns 0 in edge case where all values are 0
    """
    return (fp + fn) / (tp + tn + fp + fn + epsilon)


def calc_f1(precision, recall, epsilon=EPSILON):
    """
    Returns 0 in edge case where precision and recall are 0
    """
    return (2 * precision * recall) / (precision + recall + epsilon)


def calc_specificity(tn, fp, epsilon=EPSILON):
    """
    Returns 1 in edge case where fp=0 because there were no incorrect predictions
    """
    return (tn + epsilon) / (tn + fp + epsilon)

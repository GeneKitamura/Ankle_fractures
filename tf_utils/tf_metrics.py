import tensorflow as tf
import numpy as np


def accuracy_c_logits (logits, labels):
    return tf.reduce_mean(tf.argmax(logits, axis=1) == labels)

def model_metrics(predictions, labels):
    n = predictions.shape[0]
    predictions = np.int64(predictions)
    labels = np.int64(labels)
    zero = np.int64([0])
    _prediction_positive = np.greater(predictions, zero)
    _prediction_negative = np.equal(predictions, zero)
    _labels_positive = np.greater(labels, zero)
    _labels_negative = np.equal(labels, zero)

    # must use logical_and instead of equal to prevent situation of False == False being True
    # creating variables to allow user to keep track of false cases (would be interesting to see what types of cases were false)
    _true_positives_idx = np.logical_and(_prediction_positive, _labels_positive)
    _true_negatives_idx = np.logical_and(_prediction_negative, _labels_negative)
    _false_positives_idx = np.logical_and(_prediction_positive, _labels_negative)
    _false_negatives_idx = np.logical_and(_prediction_negative, _labels_positive)

    # You have to cast to tf.float32 before doing reduce_mean or reduce_sum
    _true_positive_num = np.sum(np.float32(_true_positives_idx))
    _true_negative_num = np.sum(np.float32(_true_negatives_idx))
    _false_positive_num = np.sum(np.float32(_false_positives_idx))
    _false_negative_num = np.sum(np.float32(_false_negatives_idx))
    total_cases = (_true_positive_num + _true_negative_num + _false_positive_num + _false_negative_num)

    sensitivity = _true_positive_num / (_true_positive_num + _false_negative_num + 1e-5)
    specificity = _true_negative_num / (_true_negative_num + _false_positive_num + 1e-5)
    PPV = _true_positive_num / (_true_positive_num + _false_positive_num + 1e-5)
    NPV = _true_negative_num / (_true_negative_num + _false_negative_num + 1e-5)
    accuracy = (_true_positive_num + _true_negative_num) / (_true_positive_num + _true_negative_num + _false_negative_num + _false_positive_num)

    return sensitivity, specificity, PPV, NPV, accuracy, _false_positives_idx, _false_negatives_idx, total_cases

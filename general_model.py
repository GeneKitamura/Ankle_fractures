#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import os
from tf_utils.tf_models import truncated_three_inception, resnet_with_auxiliary, Xception_with_auxiliary, Xception, single_resnet
from tf_utils.tf_data import _strict_data, overfitting_data
from tf_utils.tf_functions import data_augmentation, view_merging
from tf_utils.tf_metrics import model_metrics

def main():

    MODEL = Xception_with_auxiliary
    DATA_GET = overfitting_data
    AUXILIARY_AND_DROPOUT = True
    LR = 6e-6
    L2_DECAY = 0.2
    EPOCHS = 5
    BATCH_SZ = 12
    VAL_TEST_MULTIPLE = 4
    X_HEIGHT = 300
    X_WIDTH = 300
    X_CHANNELS = 1
    SHUFFLE_NUM = 3000
    N_LABELS = 2
    VALIDATE_PERIOD = 10 #keep at 10 for consistency
    TRAIN_SUMMARY = False
    VAL_SUMMARY = False
    TEST_SUMMARY = False
    SAVING_RATE = 500
    SAVING = False
    VALIDATE = True
    TEST = True
    X_LOC = None
    Y_LOC = None

    if AUXILIARY_AND_DROPOUT:
        AUX_RATE = 0.4
        DROPOUT = 0.5

    if AUXILIARY_AND_DROPOUT:
        ROOT_DIR = 'models/LR_{}_WD_{}_A_{}_D_{}_E_{}_V_{}'.format(LR, L2_DECAY, AUX_RATE, DROPOUT, EPOCHS, BATCH_SZ * VAL_TEST_MULTIPLE)
    else:
        ROOT_DIR = 'models/LR_{}_WD_{}_E_{}_V_{}'.format(LR, L2_DECAY, EPOCHS, BATCH_SZ * VAL_TEST_MULTIPLE)

    # Make the directories to save information
    os.makedirs(ROOT_DIR)
    os.mkdir("{}/models".format(ROOT_DIR))

    #Load the data into memory
    with tf.device('/cpu:0'):
        x_train, y_train, x_val, y_val, x_test, y_test = DATA_GET(BATCH_SZ, VAL_TEST_MULTIPLE, X_LOC, Y_LOC)
        n_train = x_train.shape[0]
        val_n = x_val.shape[0]
        test_n = x_test.shape[0]

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        # Creating placeholders for the input data
        x_train_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
        y_train_placeholder = tf.placeholder(y_train.dtype, y_train.shape)
        x_val_placeholder = tf.placeholder(x_val.dtype, x_val.shape)
        y_val_placeholder = tf.placeholder(y_val.dtype, y_val.shape)
        x_test_placeholder = tf.placeholder(x_test.dtype, x_test.shape)
        y_test_placeholder = tf.placeholder(y_test.dtype, y_test.shape)

        # Creating TF Databases. Train_dataset will be permutated and merged, while the val and test sets are just merged.
        train_dataset = Dataset.from_tensor_slices((x_train_placeholder, y_train_placeholder)).map(data_augmentation, num_threads=8).shuffle(SHUFFLE_NUM).batch(BATCH_SZ)

        val_dataset = Dataset.from_tensor_slices((x_val_placeholder, y_val_placeholder)).map(view_merging, num_threads=8).shuffle(SHUFFLE_NUM).batch(BATCH_SZ)

        test_dataset = Dataset.from_tensor_slices((x_test_placeholder, y_test_placeholder)).map(view_merging, num_threads=8).batch(BATCH_SZ)

        my_train_iterator = train_dataset.make_initializable_iterator()
        train_next = my_train_iterator.get_next()
        my_val_iterator = val_dataset.make_initializable_iterator()
        val_next = my_val_iterator.get_next()
        my_test_iterator = test_dataset.make_initializable_iterator()
        test_next = my_test_iterator.get_next()

        x = tf.placeholder(tf.float32, [None, X_HEIGHT, X_WIDTH, X_CHANNELS], name='data_in')
        y = tf.placeholder(tf.int64, [None], name='labels')
        is_training = tf.placeholder(tf.bool)

        if AUXILIARY_AND_DROPOUT:
            y_out, aux_out = MODEL(x, is_training=is_training, n_classes=N_LABELS, weight_decay=L2_DECAY, keeping_rate=DROPOUT)

        else:
            y_out = MODEL(x, is_training=is_training, n_classes=N_LABELS, weight_decay=L2_DECAY)

        predictions = tf.argmax(y_out, axis=1)
        correct_bool = tf.cast(tf.equal(predictions, y), tf.float32)
        correct = tf.reduce_sum(correct_bool)
        accuracy = tf.reduce_mean(correct_bool, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_out), name='cross_entropy')
        total_cross_entropy = tf.add(cross_entropy, tf.reduce_sum(tf.get_collection('l2_loss')), name='total_cross_entropy') #adding in L2 loss from collection of 'l2_loss'
        if AUXILIARY_AND_DROPOUT:
            aux_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=aux_out), name='aux_entropy')
            tf.summary.scalar('aux_entropy', aux_entropy)
            with tf.name_scope('total_loss'):
                total_loss = total_cross_entropy + AUX_RATE * aux_entropy
        else:
            total_loss = total_cross_entropy

        tf.summary.scalar('total_cross_entropy', total_cross_entropy)
        tf.summary.scalar('total_loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.9, beta2=0.999, epsilon=1e-8) # Epsilon of 0.1 good for Inception network


        # batch normalization in tensorflow requires this extra dependency - for updating moving averages and variances
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(total_loss, global_step=tf.Variable(0, trainable=False, name='global_step')) #defines the global step as number of times train_step is called

        summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(ROOT_DIR, graph=graph, filename_suffix='Train')
        val_writer = tf.summary.FileWriter(ROOT_DIR, graph=graph, filename_suffix='Val')
        test_writer = tf.summary.FileWriter(ROOT_DIR, graph=graph, filename_suffix='Test')

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init)

    #Training the model:
    train_loss_log = []
    train_accuracy_log = []
    val_loss_log = []
    val_accuracy_log = []

    for i in range(EPOCHS):
        sess.run(my_train_iterator.initializer, feed_dict={x_train_placeholder: x_train, y_train_placeholder: y_train})
        loss_i = 0
        accuracy_i = 0
        correct_i = 0

        while True:
            try:
                x_now, y_now = sess.run(train_next)

                # the feed_dict needs to encompass elements of the tf.placeholder
                _, train_loss, train_correct, train_summaries  = sess.run([train_step, total_loss, correct, summaries], feed_dict={x: x_now, y: y_now, is_training: True})

                loss_i += train_loss
                correct_i += train_correct
                if TRAIN_SUMMARY:
                    train_writer.add_summary(train_summaries, global_step=sess.run(tf.train.get_global_step(graph=graph)))

            except tf.errors.OutOfRangeError:
                break

        accuracy_i = correct_i/n_train
        train_accuracy_log.append(accuracy_i)
        train_loss_log.append(loss_i)

        print('End of epoch {0}: training loss is {1: .3g} with accuracy of {2: .3g}'.format(i, loss_i, accuracy_i))


        if VALIDATE & (i % VALIDATE_PERIOD == 0):
            sess.run(my_val_iterator.initializer, feed_dict={x_val_placeholder: x_val, y_val_placeholder: y_val})
            val_loss = 0
            val_correct = 0

            while True:
                try:
                    val_x_now, val_y_now = sess.run(val_next)
                    val_loss_k, val_correct_k, val_summaries = sess.run([total_loss, correct, summaries], feed_dict={x: val_x_now, y: val_y_now, is_training: False})
                    val_loss += val_loss_k
                    val_correct += val_correct_k

                    if VAL_SUMMARY:
                        val_writer.add_summary(val_summaries, global_step=sess.run(tf.train.get_global_step(graph=graph)))

                except tf.errors.OutOfRangeError:
                    break

            val_accuracy = val_correct / val_n
            val_loss_log.append(val_loss)
            val_accuracy_log.append(val_accuracy)
            print('VALIDATION: epoch {0}: loss is {1: .3g} with accuracy of {2: .3g} for case size of {3}'.format(i, val_loss, val_accuracy, val_n))

        if SAVING & (i % SAVING_RATE == 0):
            saver.save(sess, ROOT_DIR + '/models/Adam', global_step=sess.run(tf.train.get_global_step(graph=graph)))
            print('model saved at epoch of: ', i)


    print("Done training")

    # save the model at the very end
    if SAVING:
        saver.save(sess, ROOT_DIR + '/models/Adam', global_step=sess.run(tf.train.get_global_step(graph=graph)))

        np.save(ROOT_DIR + '/train_loss', train_loss_log)
        np.save(ROOT_DIR + '/train_accuracy', train_accuracy_log)
        np.save(ROOT_DIR + '/val_loss', val_loss_log)
        np.save(ROOT_DIR + '/val_accuracy', val_accuracy_log)

    #Test the Model:
    if TEST:
        sess.run(my_test_iterator.initializer, feed_dict={x_test_placeholder:x_test, y_test_placeholder: y_test})
        test_loss = 0
        test_correct = 0
        test_predictions = np.zeros((test_n))
        m = 0

        while True:
            try:
                x_test_now, y_test_now = sess.run(test_next)
                test_loss_m, test_correct_m, test_prediction_m, test_summaries = sess.run([total_loss, correct, predictions, summaries], feed_dict={x: x_test_now, y: y_test_now, is_training: False})
                test_loss += test_loss_m
                test_correct += test_correct_m
                test_predictions[m * BATCH_SZ: (m + 1) * BATCH_SZ] = test_prediction_m
                m += 1

                if TEST_SUMMARY:
                    test_writer.add_summary(test_summaries, global_step=sess.run(tf.train.get_global_step(graph=graph)))

            except tf.errors.OutOfRangeError:
                break

        test_accuracy = test_correct / test_n

        print("Testing the model")
        print('Test loss is {0:.3g} with test accuracy of {1:.3g}'.format(test_loss, test_accuracy))

        np.save(ROOT_DIR + '/test_accuracy', test_accuracy)
        np.save(ROOT_DIR + '/test_loss', test_loss)

        sensitivity, specificity, PPV, NPV, _, _false_positives_idx, _false_negatives_idx, total_test_cases = model_metrics(test_predictions, y_test)

        print('For the test set: sensitivity/specificity are {0:.3g} / {1:.3g}.  The PPV/NPV are {2:.3g} / {3:.3g}.'.format(sensitivity, specificity, PPV, NPV))
        print('total test cases: ', total_test_cases)

        np.save(ROOT_DIR + "/test_sensitivity", sensitivity)
        np.save(ROOT_DIR + "/test_specificity", specificity)
        np.save(ROOT_DIR + "/test_PPV", PPV)
        np.save(ROOT_DIR + "/test_NPV", NPV)
        np.save(ROOT_DIR + "/test_false_positives_idx", _false_positives_idx)
        np.save(ROOT_DIR + "/test_false_negatives_idx", _false_negatives_idx)

if __name__ == '__main__':
    main()

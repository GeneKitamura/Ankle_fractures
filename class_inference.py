import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tf_utils.tf_functions import view_merging
from tf_utils.tf_metrics import model_metrics
from tf_utils.tf_models import resnet_with_auxiliary, single_resnet, Xception, Xception_with_auxiliary, truncated_three_inception
import numpy as np
import matplotlib.pyplot as plt
import math

class Models(object):
    def __init__(self, X_HEIGHT, X_WIDTH, X_CHANNELS, N_LABELS, L2_DECAY, AUX, DROPOUT, MODEL, BATCH_SZ):
        self.height = X_HEIGHT
        self.width = X_WIDTH
        self.channels = X_CHANNELS
        self.n_labels = N_LABELS
        self.L2_decay = L2_DECAY
        self.aux = AUX
        self.dropout = DROPOUT
        self.model = MODEL
        self.batch_sz = BATCH_SZ


    def create_sess_graph(self, restore_path):
        graph = tf.Graph()
        with graph.as_default():

            self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='data_in')
            self.y = tf.placeholder(tf.int64, [None], name='labels')
            self.is_training = tf.placeholder(tf.bool, name='q_training')
            self.x_test_placeholder = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='x_test_holder')
            self.y_test_placeholder = tf.placeholder(tf.int64, [None], name='y_test_holder')

            if self.aux:
                y_out, aux_out = self.model(self.x, is_training=self.is_training, n_classes=self.n_labels, weight_decay=self.L2_decay, keeping_rate=self.dropout)

            else:
                y_out = self.model(self.x, is_training=self.is_training, n_classes=self.n_labels, weight_decay=self.L2_decay)

            self.predictions = tf.argmax(y_out, axis=1, name='predictions')
            correct_bool = tf.cast(tf.equal(self.predictions, self.y), tf.float32)
            self.correct = tf.reduce_sum(correct_bool, name='correct')

            saver = tf.train.Saver()

        sess = tf.Session(graph=graph)
        saver.restore(sess, restore_path)

        self.sess = sess
        self.graph = graph #should be with the graph with restored VARIABLES

    def test_model(self, x_test, y_test, return_accuracy = False, print_out=True, voting=True):
        with self.graph.as_default():

            test_dataset = Dataset.from_tensor_slices((self.x_test_placeholder, self.y_test_placeholder)).map(view_merging, num_threads=8).batch(self.batch_sz)

            my_test_iterator = test_dataset.make_initializable_iterator()
            test_next = my_test_iterator.get_next()
            test_n = x_test.shape[0]

            self.sess.run(my_test_iterator.initializer, feed_dict={self.x_test_placeholder: x_test, self.y_test_placeholder: y_test})
            test_correct = 0
            test_predictions = np.zeros((test_n))
            m = 0

            while True:
                try:
                    x_test_now, y_test_now = self.sess.run(test_next)
                    test_correct_m, test_prediction_m = self.sess.run([self.correct, self.predictions],feed_dict={self.x: x_test_now, self.y: y_test_now, self.is_training: False})
                    test_correct += test_correct_m
                    test_predictions[m * self.batch_sz: (m + 1) * self.batch_sz] = test_prediction_m
                    m += 1

                except tf.errors.OutOfRangeError:
                    break

            test_accuracy = test_correct / test_n

            if print_out:
                print("Testing the model")
                print('Test accuracy of {0:.3g}'.format(test_accuracy))

                sensitivity, specificity, PPV, NPV, _, _false_positives_idx, _false_negatives_idx, total_test_cases = model_metrics(test_predictions, y_test)

                print('For the test set: sensitivity/specificity are {0:.3g} / {1:.3g}.  The PPV/NPV are {2:.3g} / {3:.3g}.'.format(sensitivity, specificity, PPV, NPV))
                print('total test cases: ', total_test_cases)

            if voting:
                print('Voting now')
                self.predictions =  test_predictions

            if return_accuracy:
                print('Test accuracy of {0:.3g}'.format(test_accuracy))
                return test_accuracy

    def do_and_undo_conv1(self, tf_op_name, samples, scope='conv1', ask_training=False):
        with self.graph.as_default():
            # values method on tf_ops (void function) return the tensor of interest.
            # alternatively can use tf_tensor, which is going to be tf_op:0, and won't need the values method or unwrapping
            activation_layer =  self.sess.run(self.graph.get_operation_by_name(scope + tf_op_name).values(), feed_dict={self.x:samples, self.is_training: ask_training})[0]
            # need to unwrap with [0] since values method returns a tuple

            # Getting the necessary variables and creating graph elements
            with tf.variable_scope(scope, reuse=True):
                conv1_kernel = tf.get_variable('kernel')
                conv1_bias = tf.get_variable('bias')
                conv1_bn_beta = tf.get_variable('bn/beta')
                conv1_bn_gamma = tf.get_variable('bn/gamma')
                conv1_bn_moving_mean = tf.get_variable('bn/moving_mean')
                conv1_bn_moving_variance = tf.get_variable('bn/moving_variance')


            # checking to see if first conv padding was valid or same.
            # For images of 300 x 300, valid padding conv2d with strides of 2 results in shape of 149 x 149, while same padding results in 150 x 150
            # even though output shape is specified as a parameter in conv2d_transpose, the expected input shape is different when the pad is 'SAME' or 'VALID'.
            # For example, with resnet, the conv1 output shape is 147 x 147.  When doing conv2d_transpose with expected output of 300 x 300, if padding is 'VALID', expected input shape is 147 x 147, but when the padding is the 'SAME', the expected input shape is 150 x 150.
            if activation_layer.shape[1] == self.height / 2:
                pad_choice = 'SAME'
            else:
                pad_choice = 'VALID'


            # Building the tensorflow graph for image reconstruction
            # In paper, describes reversing max-pool (which I don't have in the first conv layers).
            # They did not unto the bn and bias, but I do here
            # Works with SINGLE sample, as I need it to work for both N and a single sample
            # input size from 1st activation is variable
            curr_inp = tf.placeholder(np.float32, [1, activation_layer.shape[1], activation_layer.shape[2], activation_layer.shape[3]], name='upstream_x')
            un_relu = tf.nn.relu(curr_inp, name='un_relu')
            with tf.name_scope('un_bn'):
                un_bn = (((un_relu - conv1_bn_beta) * conv1_bn_moving_variance) / conv1_bn_gamma) + conv1_bn_moving_mean
            with tf.name_scope('un_bias'):
                un_bias = un_bn - conv1_bias
            # the conv2d_transpose filter in-out channels are flipped compared to normal conv2d
            un_conv1 = tf.nn.conv2d_transpose(un_bias, conv1_kernel, output_shape=[1, self.height, self.width, self.channels], strides=[1, 2, 2, 1], padding=pad_choice, name='un_conv1')

            self.un_conv1 = un_conv1
            self.unconv_inp = curr_inp

            return activation_layer

    # takes in N x H x W x C image, which for us is either N x 149 x 149 x 32 or N x 150 x 150 x 1
    def imageify_unconv1(self, inp_images, full_FM_recon_sample):

        assert(self.graph is not None), "Run create_sess_graph method first"
        assert(self.un_conv1 is not None), "Run do_and_undo_conv1 method first"

        inp_N = inp_images.shape[0]
        FM_num = inp_images.shape[3]

        # reconstruction from 1st layer using all FM for all samples
        tot_out = np.zeros((inp_N, self.height, self.width, self.channels))
        for i in range(inp_N):
            tmp = np.expand_dims(inp_images[i], axis=0) # adding first dims
            tot_out[i] = self.sess.run(self.un_conv1, feed_dict={self.unconv_inp: tmp})

        # For a single sample, using each of the feature_map to reconstruct the whole image
        one_image_FM_recon = np.zeros([FM_num, self.height, self.width, self.channels])
        act_image = np.expand_dims(inp_images[full_FM_recon_sample], axis=0) # 1 x 150 x 150 x 32

        for i in range(FM_num):
            hot_feature_map = np.zeros((act_image.shape)) # array of zeros to put in a single FM into
            hot_feature_map[...,i] = act_image[..., i] # filling in a single FM

            one_image_FM_recon[i] = self.sess.run(self.un_conv1, feed_dict={self.unconv_inp:hot_feature_map})

        return tot_out, one_image_FM_recon

    @staticmethod
    def show_featuremap_recons(feature_map_array, show_title=False, save=False, height=20, width=20, format='eps', dpi=900):

        # creates a 'figure' object and array of 'axes' objects.
        # The subplot_kw is a dictionary (with CONFUSING kwargs).
        # Additional kwarg** are passed into the figure object (such as figsize).
        FM_num = feature_map_array.shape[0]
        n_slots = int(math.ceil(np.sqrt(FM_num)))

        fig, axes = plt.subplots(n_slots, n_slots, subplot_kw= {}, figsize=(height, width))
        fig.subplots_adjust(hspace=0, wspace=0)

        feature_map_n = feature_map_array.shape[0]

        for ax, i in zip(axes.flatten(), range(feature_map_n)):
            ax.imshow(feature_map_array[i, ..., 0], cmap=plt.cm.gray)
            if show_title:
                ax.set_title("FM# : {}".format(i+1), fontdict={})
            ax.axis('off')

        if save:
            plt.savefig('./plots/recons', format=format, dpi=dpi)

        plt.show()

    @staticmethod
    def show_activations(activations_array, sample_num, show_title=False, save=False, height=20, width=20, format='eps', dpi=900):
        FM_num = activations_array.shape[3]
        n_slots = int(math.ceil(np.sqrt(FM_num)))

        fig, axes = plt.subplots(n_slots, n_slots, subplot_kw= {}, figsize=(height, width))
        fig.subplots_adjust(hspace=0, wspace=0)

        feature_map_n = activations_array.shape[3]

        for ax, i in zip(axes.flatten(), range(feature_map_n)):
            ax.imshow(activations_array[sample_num, ..., i], cmap=plt.cm.gray)
            if show_title:
                ax.set_title("FM# : {}".format(i+1), fontdict={})
            ax.axis('off')

        if save:
            plt.savefig('./plots/activation_num_{}'.format(sample_num), format=format, dpi=dpi)

        plt.show()


class Single_Models(Models):
    def __init__(self, x_height, x_width, x_channels, n_labels, L2_decay, aux, dropout, model, batch_num):
        super().__init__(x_height, x_width, x_channels, n_labels, L2_decay, aux, dropout, model, batch_num)

    @classmethod
    def simple_resnet(cls):
        return cls(300, 300, 1, 2, None, False, None, single_resnet, 12)

    @classmethod
    # dropout rate doesn't matter with tf.layers
    def resnet_aux(cls):
        return cls(300, 300, 1, 2, None, True, 0.5, resnet_with_auxiliary, 12)

    @classmethod
    def simple_Xception(cls):
        return cls(300, 300, 1, 2, None, False, None, Xception, 20)

    @classmethod
    # dropout rate doesn't matter with tf.layers
    def Xception_aux(cls):
        return cls(300, 300, 1, 2, None, True, 0.5, Xception_with_auxiliary, 20)

    @classmethod
    # dropout NEEDS to be at one because the model used tf.nn after than tf.layers
    def Inception(cls):
        return cls(300, 300, 1, 2, None, True, 1.0, truncated_three_inception, 30)
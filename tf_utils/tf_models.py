import tensorflow as tf
import numpy as np
from tf_utils.model_functions import init_weight, init_bias, conv_bn_relu, bn_relu_conv, projection, _add_units, double_separable_pool_and_add, triple_separable_and_add, simple_separable_relu, auxillary_classifier
from tf_utils.tf_properties import add_collection, activation_summary



#When images are size 299 x 299 x 1 or 300 x 300 x 1.  Results in the same shape after fist conv layer with 'VALID' shape.
def truncated_three_inception(x, is_training, n_classes, weight_decay, keeping_rate=0.5):

    def _inception_1(data_in, f_1x1, _f_1x1_2, f_3x3, _f_1x1_3, _f_3x3, f_5x5, f_pool): # with average_pooling
        _, _, _, Fi = data_in.shape
        conv_1x1_1 = conv_bn_relu(data_in, [1, 1, Fi, f_1x1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x1_1', training=is_training)

        _conv_1x1_2 = conv_bn_relu(data_in, [1, 1, Fi, _f_1x1_2], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1_2', training=is_training)
        conv_3x3 = conv_bn_relu(_conv_1x1_2, [3, 3, _f_1x1_2, f_3x3], strides=[1, 1, 1, 1], padding='SAME', name='conv_3x3', training=is_training)

        _conv_1x1_3 = conv_bn_relu(data_in, [1, 1, Fi, _f_1x1_3], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x2_3', training=is_training)
        _conv_3x3 = conv_bn_relu(_conv_1x1_3, [3, 3, _f_1x1_3, _f_3x3], strides=[1, 1, 1, 1], padding='SAME', name='_conv_3x3', training=is_training)
        conv_5x5 = conv_bn_relu(_conv_3x3, [3, 3, _f_3x3, f_5x5], strides=[1, 1, 1, 1], padding='SAME', name='conv_5x5', training=is_training)

        _pool = tf.nn.avg_pool(data_in, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='_pool')
        _, _, _, F = _pool.shape
        pool_conv = conv_bn_relu(_pool, [1, 1, F, f_pool], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool', training=is_training)

        return tf.concat([conv_1x1_1, conv_3x3, conv_5x5, pool_conv], axis=3, name='concat') # concate at axis 3 since the features at the end

    def _inception_7(data_in, f_1x1_1, _f_1x1_2, _f_1x7_2, f_7x1_2, _f_1x1_3, _f_1x7_3_first, _f_7x1_3_first, _f_1x7_3_second, f_7x1_3_second, f_pool):
        _, _, _, Fi = data_in.shape
        conv_1x1_1 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, f_1x1_1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x1_1', training=is_training)

        _conv_1x1_2 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, _f_1x1_2], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1_2', training=is_training)
        _conv_1x7_2 = conv_bn_relu(_conv_1x1_2, kernel_size=[1, 7, _f_1x1_2, _f_1x7_2], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x7_2', training=is_training)
        conv_7x1_2 = conv_bn_relu(_conv_1x7_2, kernel_size=[7, 1, _f_1x7_2, f_7x1_2], strides=[1, 1, 1, 1], padding='SAME', name='conv_7x1_2', training=is_training)

        _conv_1x1_3 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, _f_1x1_3], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1_3', training=is_training)
        _conv_1x7_3_first = conv_bn_relu(_conv_1x1_3, kernel_size=[1, 7, _f_1x1_3, _f_1x7_3_first], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x7_3_first', training=is_training)
        _conv_7x1_3_first = conv_bn_relu(_conv_1x7_3_first, kernel_size=[7, 1, _f_1x7_3_first, _f_7x1_3_first], strides=[1, 1, 1, 1], padding='SAME', name='_conv_7x1_3_first', training=is_training)
        _conv_1x7_3_second = conv_bn_relu(_conv_7x1_3_first, kernel_size=[1, 7, _f_7x1_3_first, _f_1x7_3_second], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x7_3_second', training=is_training)
        conv_7x1_3_second = conv_bn_relu(_conv_1x7_3_second, kernel_size=[7, 1, _f_1x7_3_second, f_7x1_3_second], strides=[1, 1, 1, 1], padding='SAME', name='conv_7x1_3_second', training=is_training)

        _avg_pool = tf.nn.avg_pool(data_in, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='_avg_pool')
        _, _, _, F = _avg_pool.shape
        pool_conv = conv_bn_relu(_avg_pool, kernel_size=[1, 1, F, f_pool], strides=[1, 1, 1, 1], padding='SAME', name='pool_conv', training=is_training)

        return tf.concat([conv_1x1_1, conv_7x1_2, conv_7x1_3_second, pool_conv], axis=3, name='concat')

    def auxillary_classifier(data_in, fc_1, fc_2):
        # data_in should be size 17 x 17 x 768, as it comes at the end of the last _inception_7 layer.
        # fc_1 and fc_2 refers to the size of the FC output.

        with tf.variable_scope('auxillary_classifier'):
            _avg_pool = tf.nn.avg_pool(data_in, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID', name='_avg_pool') #size of 5 x 5 x 768
            assert (_avg_pool.shape[1:] == (5, 5, 768)), "Shape of aux_avg_pool is incorrect."
            _, _, _, F = _avg_pool.shape
            _conv_1x1 = conv_bn_relu(_avg_pool, kernel_size=[1, 1, F, 128], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1', training=is_training) # size of 5 x 13 x 128
            with tf.name_scope('conv_1x1'):
                activation_summary(_conv_1x1)

            N, H, W, F = _conv_1x1.shape
            conv_flattened = tf.reshape(_conv_1x1, [-1, int(H*W*F)], name='conv_flattened')
            N, D = conv_flattened.shape
            FC_W1 = init_weight('FC_W1', shape=[D, fc_1], alpha=weight_decay) #fc_1 is hyperparameter; pick maybe ~700?
            FC_b1 = init_bias('FC_b1', shape=fc_1)
            FC_1_NN_relu = tf.nn.relu(tf.matmul(conv_flattened, FC_W1) + FC_b1, name='FC_1_NN_relu')
            with tf.name_scope('FC_1_NN_relu'):
                activation_summary(FC_1_NN_relu)

            FC_W2 = init_weight('FC_W2', shape=[fc_1, fc_2], alpha=weight_decay) # fc_2 is the final output class number; for binary cases, should be 2
            FC_b2 = init_bias('FC_b2', shape=[fc_2])
            FC_2_NN = tf.matmul(FC_1_NN_relu, FC_W2) + FC_b2 #size is N x 2
            with tf.name_scope('FC_2_NN'):
                activation_summary(FC_2_NN)

            return FC_2_NN # before the softmax; just the logits.  The losses need to be weighted; in paper, is was weighted by 0.3.

    def _inception_8 (data_in, f_1x1_1, _f_1x1_2, f_1x3_1, f_3x1_1, _f_1x1_3, _f_3x3, _f_1x3_2, _f_3x1_2, _f_1x1_4):
        _, _, _, Fi = data_in.shape
        conv_1x1_1 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, f_1x1_1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x1_1', training=is_training)

        _conv_1x1_2 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, _f_1x1_2], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1_2', training=is_training)
        conv_1x3_1 = conv_bn_relu(_conv_1x1_2, kernel_size=[1, 3, _f_1x1_2, f_1x3_1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x3_1', training=is_training)
        conv_3x1_1 = conv_bn_relu(_conv_1x1_2, kernel_size=[3, 1, _f_1x1_2, f_3x1_1], strides=[1, 1, 1, 1], padding='SAME', name='conv_3x1_1', training=is_training)

        _conv_1x1_3 = conv_bn_relu(data_in, kernel_size=[1, 1, Fi, _f_1x1_3], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1_3', training=is_training)
        _conv_3x3 = conv_bn_relu(_conv_1x1_3, kernel_size=[3, 3, _f_1x1_3, _f_3x3], strides=[1, 1, 1, 1], padding='SAME', name='_conv_3x3', training=is_training)
        conv_1x3_2 = conv_bn_relu(_conv_3x3, kernel_size=[1, 3, _f_3x3, _f_1x3_2], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x3_2', training=is_training)
        conv_3x1_2 = conv_bn_relu(_conv_3x3, kernel_size=[3, 1, _f_3x3, _f_3x1_2], strides=[1, 1, 1, 1], padding='SAME', name='conv_3x1_2', training=is_training)

        _avg_pool = tf.nn.avg_pool(data_in, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool')
        _, _, _, F = _avg_pool.shape
        conv_1x1_4 = conv_bn_relu(_avg_pool, kernel_size=[1, 1, F, _f_1x1_4], strides=[1, 1, 1, 1], padding='SAME', name='conv_1x1_4', training=is_training)

        return tf.concat([conv_1x1_1, conv_1x3_1, conv_3x1_1, conv_1x3_2, conv_3x1_2, conv_1x1_4], axis=3, name='concat')

    #assumption is that the starting image size is N x 299 x 299 x 1
    #equation for calculating shape-out is o = [i + 2*pad - kernel]/strides + 1

    # first conv layer with BN and relu
    with tf.variable_scope('conv1'):
        conv1_out = conv_bn_relu(x, kernel_size=[3, 3, 1, 32], strides=[1, 2, 2, 1], padding='VALID', name='out', training=is_training) # size is 149 x 149 x 32
        activation_summary(conv1_out)
        add_collection('conv1', conv1_out)

    # second conv layer with BN and relu
    with tf.variable_scope('conv2'):
        conv2_out = conv_bn_relu(conv1_out, kernel_size=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding='VALID', name='out', training=is_training) # size is 147 x 147 x 32
        activation_summary(conv2_out)
        add_collection('conv2', conv2_out)

    # third conv layer with BN and relu
    with tf.variable_scope('conv3'):
        conv3_out = conv_bn_relu(conv2_out, kernel_size=[3, 3, 32, 64], strides=[1, 1, 1, 1], padding='SAME', name='out', training=is_training) # size is 147 x 147 x 64
        activation_summary(conv3_out)
        add_collection('conv3', conv3_out)

    #first max_pool
    max_pool_1 = tf.nn.max_pool(conv3_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool_1', padding='VALID')  # 73 x 73 x 64
    assert (max_pool_1.shape[1:] == (73, 73, 64)), "Shape at max_pool_1 is incorrect at {}.".format(max_pool_1.shape[1:])
    with tf.name_scope('max_pool_1'):
        activation_summary(max_pool_1)

    # fourth conv layer
    with tf.variable_scope('conv4'):
        conv4_out = conv_bn_relu(max_pool_1, kernel_size=[1, 1, 64, 80], strides=[1, 1, 1, 1], padding='SAME', name='out', training=is_training) #73 x 73 x 80
        activation_summary(conv4_out)

    # fifth conv layer
    with tf.variable_scope('conv5'):
        conv5_out = conv_bn_relu(conv4_out, kernel_size=[3, 3, 80, 192], strides=[1, 1, 1, 1], padding='VALID', name='out', training=is_training) # 71 x 71 x 192
        activation_summary(conv5_out)

    # second max_pooling
    max_pool_2 = tf.nn.max_pool(conv5_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool_2', padding='VALID')  # size is 35 x 35 x 192
    assert (max_pool_2.shape[1:] == (35, 35, 192)), "Shape at max_pool_2 is incorrect at {}".format(max_pool_2.shape[1:])
    with tf.name_scope('max_pool_2'):
        activation_summary(max_pool_2)

    #first 5x5 inception layer
    with tf.variable_scope('inception_5x5_1'):
        inception_5x5_1 = _inception_1(max_pool_2, 64, 48, 64, 64, 96, 96, 32) # size is 35 x 35 x 192
        activation_summary(inception_5x5_1)

    #second 5x5 inception layer
    with tf.variable_scope('inception_5x5_2'):
        inception_5x5_2 = _inception_1(inception_5x5_1, 64, 48, 64, 64, 96, 96, 64) # size is 35 x 35 x 288
        activation_summary(inception_5x5_2)

    #third 5x5 inception layer
    with tf.variable_scope('inception_5x5_3'):
        inception_5x5_3 = _inception_1(inception_5x5_2, 64, 48, 64, 64, 96, 96, 64) # size is still 35 x 35 x 288
        activation_summary(inception_5x5_3)

    # short layer between the 5x5 and 7x7 layers
    with tf.variable_scope('short_layer_1'):
        short_1_conv = conv_bn_relu(inception_5x5_3, kernel_size=[3, 3, 288, 384], strides=[1, 2, 2, 1], padding='VALID', name='conv_1x1', training=is_training)
        short_1_pool = tf.nn.max_pool(inception_5x5_3, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='VALID', name='pool')
        _short_1_1x1 = conv_bn_relu(inception_5x5_3, kernel_size=[1, 1, 288, 64], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1', training=is_training)
        _short_1_3x3 = conv_bn_relu(_short_1_1x1, kernel_size=[3, 3, 64, 96], strides=[1, 1, 1, 1], padding='SAME', name='_conv_3x3', training=is_training)
        short_1_5x5 = conv_bn_relu(_short_1_3x3, kernel_size=[3, 3, 96, 96], strides=[1, 2, 2, 1], padding='VALID', name='conv_5x5', training=is_training)
        short_1_out = tf.concat([short_1_conv, short_1_5x5, short_1_pool], axis=3, name='out')  #should be size 17 x 17 x 768
        assert (short_1_out.shape[1:] == (17, 17, 768))
        activation_summary(short_1_out)

    #first 7x7 inception layer
    with tf.variable_scope('inception_7_1'):
        inception_7_1 = _inception_7(short_1_out, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192) # size is 17 x 17 x 768
        activation_summary(inception_7_1)

    #second 7x7 inception layer
    with tf.variable_scope('inception_7_2'):
        inception_7_2 = _inception_7(inception_7_1, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192) # size is 17 x 17 x 768
        activation_summary(inception_7_2)

    #third 7x7 inception layer
    with tf.variable_scope('inception_7_3'):
        inception_7_3 = _inception_7(inception_7_2, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192) # size is 17 x 17 x 768
        activation_summary(inception_7_3)

    #fourth 7x7 inception layer
    with tf.variable_scope('inception_7_4'):
        inception_7_4 = _inception_7(inception_7_3, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192) # size is 17 x 17 x 768
        activation_summary(inception_7_4)

    #Auxially classifier
    auxillary_out = auxillary_classifier(inception_7_4, 700, n_classes)

    # back to the incpetion network
    # second short layer with input size 17 x 17 x 768
    with tf.variable_scope('short_layer_2'):
        short_2_pool = tf.nn.max_pool(inception_7_4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')
        _short_2_1x1_1 = conv_bn_relu(inception_7_4, kernel_size=[1, 1, 768, 192], strides=[1, 1, 1, 1], padding='SAME', name='_1x1_1', training=is_training)
        short_2_3x3 = conv_bn_relu(_short_2_1x1_1, kernel_size=[3, 3, 192, 320], strides=[1, 2, 2, 1], padding='VALID', name='3x3', training=is_training)
        _short_2_1x1_2 = conv_bn_relu(inception_7_4, kernel_size=[1, 1, 768, 192], strides=[1, 1, 1, 1], padding='SAME', name='_1x1_2', training=is_training)
        _short_2_1x7 = conv_bn_relu(_short_2_1x1_2, kernel_size=[1, 7, 192, 192], strides=[1, 1, 1, 1], padding='SAME', name='_1x7', training=is_training)
        _short_2_7x1 = conv_bn_relu(_short_2_1x7, kernel_size=[7, 1, 192, 192], strides=[1, 1, 1, 1], padding='SAME', name='_7x1', training=is_training)
        short_2_7x7 = conv_bn_relu(_short_2_7x1, kernel_size=[3, 3, 192, 192], strides=[1, 2, 2, 1], padding='VALID', name='7x7', training=is_training)
        short_2_out = tf.concat([short_2_3x3, short_2_7x7, short_2_pool], axis=3, name='out') # should now be size 8 x 8 x 1280
        assert (short_2_out.shape[1:] == (8 , 8 , 1280)), "Short_2_out shape is incorrect at {}".format(short_2_out.shape[1:])
        activation_summary(short_2_out)

    with tf.variable_scope('inception_8_1'):
        inception_8_1 = _inception_8(short_2_out, 320, 384, 384, 384, 448, 384, 384, 384, 192) # output should be 8 x 8 x 2048
        activation_summary(inception_8_1)

    with tf.variable_scope('inception_8_2'):
        inception_8_2 = _inception_8(inception_8_1, 320, 384, 384, 384, 448, 384, 384, 384, 192) # output should be 8 x 8 x 2048
        activation_summary(inception_8_2)

    with tf.variable_scope('final_output'):
        _avg_pool = tf.nn.avg_pool(inception_8_2, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='_avg_pool') #out-size is 1 x 1 x 2048
        assert (_avg_pool.shape[1:] == (1, 1, 2048)), "The shape of final_output_pool is incorrect at {}".format(_avg_pool.shape[1:])
        with tf.name_scope('avg_pool'):
            activation_summary(_avg_pool)

        _dropout = tf.nn.dropout(_avg_pool, keep_prob=keeping_rate, name='_dropout')
        with tf.name_scope('dropout'):
            activation_summary(_dropout)

        N, H, W, F = _dropout.shape
        _flattened = tf.reshape(_dropout, [-1, int(H*W*F)], name='_flattened') # size of 2048
        N, D = _flattened.shape
        FC_W = init_weight('FC_W', shape=[D, n_classes], alpha=weight_decay)
        FC_b = init_bias('FC_b', shape=n_classes)
        logits = tf.matmul(_flattened, FC_W) + FC_b
        with tf.name_scope('FC'):
            activation_summary(logits)

    return logits, auxillary_out

# When input images are size 300 x 300 x 1
def single_resnet(x, is_training, n_classes, weight_decay, depth=101):
    # IMPORTANT POINTS:
    # For pre-activation Residual Units, make sure to do RELU BEFORE the first path split/addition and AFTER the last element-wise addition.
    assert (depth == 50 or depth == 101), "Depth needs to be either 50 or 101"

    UNIT_STRIDE = [1, 1, 1, 1]
    DOUBLE_STRIDE = [1,2,2,1]
    SAME_STR = 'SAME'
    VALID_STR = 'VALID'
    LONG_REPEAT = 23 if depth == 101 else 6
    LAYER_MULTIPLES = [3, 4, LONG_REPEAT, 3]

    # Units strides with SAME padding results in NO change in feature_map sizes
    def _residual_unit(data_in, f_1x1_1, f_3x3, f_1x1_2, downsampling):
        _, _, _, Fi = data_in.shape
        conv_1x1_1 = bn_relu_conv(data_in, [1, 1, Fi, f_1x1_1], padding=SAME_STR, strides=UNIT_STRIDE, name='conv_1x1_1', training=is_training)
        Fi = conv_1x1_1.shape[3]
        selected_stride = UNIT_STRIDE if not downsampling else DOUBLE_STRIDE
        conv_3x3 = bn_relu_conv(conv_1x1_1, [3, 3, Fi, f_3x3], padding=SAME_STR, strides=selected_stride, name='conv_3x3', training=is_training)
        Fi = conv_3x3.shape[3]
        conv_1x1_2 = bn_relu_conv(conv_3x3, [1, 1, Fi, f_1x1_2], padding=SAME_STR, strides=UNIT_STRIDE, name='conv_1x1_2', training=is_training)

        return conv_1x1_2

    #equation for calculating shape-out is o = [i + 2*pad - kernel]/strides + 1
    #initial input is 300 x 300 x 1

    with tf.variable_scope('conv1'):
        conv1_out = conv_bn_relu(x, kernel_size=[7, 7, 1, 64], strides=DOUBLE_STRIDE, padding=VALID_STR, name='conv1', training=is_training) # 147 x 147 x 64

    max_pool_1 = tf.nn.max_pool(conv1_out, ksize=[1,3,3,1], strides=DOUBLE_STRIDE, name='max_pool_1', padding=SAME_STR) # 74 x 74 x 64
    assert (max_pool_1.shape[1:] == (74, 74, 64)), "Shape of max_pool_1 incorrect."

    with tf.variable_scope('max_pool_project'):
        proj_pool = projection(max_pool_1, 256, down_sample=False)
    assert (proj_pool.shape[1:] == (74, 74, 256)), "Shape of max_pool_project incorrect"

    with tf.variable_scope('conv2_1'):
        conv2_1 = _residual_unit(max_pool_1, 74, 74, 256, downsampling=False)
        prev_output = _add_units(proj_pool, conv2_1)

    #First multi-residual unit layer
    for i in range(LAYER_MULTIPLES[0]-1):
        with tf.variable_scope('conv2_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 64, 64, 256, downsampling=False)
            prev_output = _add_units(out_now, prev_output)
    assert (prev_output.shape[1:] == (74, 74, 256)), "Shape after conv2 incorrect"

    # First layer of conv3 with downsampling
    # First projection since feature_map sizes are now going to be different.
    # conv2 sizes is 74 x 74 x 256.  conv3 sizes are 37 x 37 x 512
    with tf.variable_scope('conv3_1'):
        conv3_1 = _residual_unit(prev_output, 128, 128, 512, downsampling=True) # shape is 37 x 112 x 512
        proj_2_3 = projection(prev_output, 512) #should be size 37 x 37 x 512
        prev_output = _add_units(proj_2_3, conv3_1)

    #Second multi-residual unit layer from conv3_2 on
    for i in range(LAYER_MULTIPLES[1]-1):
        with tf.variable_scope('conv3_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 128, 128, 512, downsampling=False)
            prev_output = _add_units(out_now, prev_output)
    assert (prev_output.shape[1:] == (37, 37, 512)), "Shape after conv3 incorrect"

    #2nd projection level
    with tf.variable_scope('conv4_1'):
        conv4_1 = _residual_unit(prev_output, 256, 256, 1024, downsampling=True) #size 19 x 19 x 1024
        proj_3_4 = projection(prev_output, 1024) # size 19 x 19 x 1024
        prev_output = _add_units(conv4_1, proj_3_4)

    print('3rd multi-layer depth: {}'.format(LAYER_MULTIPLES[2] -1))
    # Third multi-residual unit layer
    for i in range(LAYER_MULTIPLES[2] -1):
        with tf.variable_scope('conv4_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 256, 256, 1024, downsampling=False)
            prev_output = _add_units(out_now, prev_output) # size of 19 x 19 x 1024
    assert (prev_output.shape[1:] == (19, 19, 1024)), "Shape after conv4 incorrect"

    #3rd projection level
    with tf.variable_scope('conv5_1'):
        conv5_1 = _residual_unit(prev_output, 512, 512, 2048, downsampling=True)
        proj_4_5 = projection(prev_output, 2048)
        prev_output = _add_units(conv5_1, proj_4_5) # output shape of 10 x 10 x 2048

    # Fourth/last multi-residual unit layer
    for i in range(LAYER_MULTIPLES[3] -1):
        with tf.variable_scope('conv5_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 512, 512, 2048, downsampling=False)
            prev_output = _add_units(out_now, prev_output) # size of 10 x 10 x 2048
    assert (prev_output.shape[1:] == (10, 10, 2048)), "Shape after conv5 incorrect"


    with tf.variable_scope('final_output'):
        post_relu = tf.nn.relu(prev_output, name='post_relu')
        _avg_pool = tf.nn.avg_pool(post_relu, ksize=[1, 10, 10, 1], strides=UNIT_STRIDE, padding=VALID_STR, name='avg_pool') #outsize is now 1 x 1 x 2048
        N, H, W, F = _avg_pool.shape
        _flattened = tf.reshape(_avg_pool, [-1, int(H*W*F)], name='flattened')
        N, D = _flattened.shape
        FC_W = init_weight('FC_W', shape=[D, n_classes], alpha=weight_decay)
        FC_b = init_bias('FC_b', shape=n_classes)
        logits = tf.matmul(_flattened, FC_W) + FC_b

        return logits


# When input images are size 300 x 300 x 1
def Xception(x, is_training, n_classes, weight_decay):

    UNIT_STRIDE = [1, 1, 1, 1]
    DOUBLE_STRIDE = [1,2,2,1]
    SAME_STR = 'SAME'
    VALID_STR = 'VALID'
    LAYERS_NUM = 8

    curr_layer = conv_bn_relu(x, [3, 3, 1, 32], strides=DOUBLE_STRIDE, padding=SAME_STR, name='conv1', training=is_training)
    curr_layer = conv_bn_relu(curr_layer, [3, 3, 32, 64], strides=UNIT_STRIDE, padding=SAME_STR, name='conv2', training=is_training)

    with tf.variable_scope('double_1'):
        curr_layer = double_separable_pool_and_add(curr_layer, 128, 128, True, is_training)

    with tf.variable_scope('double_2'):
        curr_layer = double_separable_pool_and_add(curr_layer, 256, 256, False, is_training)

    with tf.variable_scope('double_3'):
        curr_layer = double_separable_pool_and_add(curr_layer, 728, 728, False, is_training)

    for i in range(LAYERS_NUM):
        with tf.variable_scope('triple_{}'.format(i+1)):
            curr_layer = triple_separable_and_add(curr_layer, 728, training=is_training)

    with tf.variable_scope('double_exit'):
        curr_layer = double_separable_pool_and_add(curr_layer, 728, 1024, False, is_training)

    with tf.variable_scope('conv3'):
        curr_layer = simple_separable_relu(curr_layer, 1536, training=is_training)

    with tf.variable_scope('conv4'):
        curr_layer = simple_separable_relu(curr_layer, 2048, training=is_training)

    with tf.variable_scope('final_output'):
        curr_layer = tf.nn.avg_pool(curr_layer, [1, 10, 10, 1], strides=UNIT_STRIDE, padding=VALID_STR, name='global_avg')
        N, H, W, F = curr_layer.shape
        _flattened = tf.reshape(curr_layer, [-1, int(H*W*F)], name='flattened')
        N, D = _flattened.shape
        FC_W = init_weight('FC_W', shape=[D, n_classes], alpha=weight_decay)
        FC_b = init_bias('FC_b', shape=n_classes)
        logits = tf.matmul(_flattened, FC_W) + FC_b

        return logits


def resnet_with_auxiliary(x, is_training, n_classes, weight_decay, depth=101, keeping_rate=0.5):
    # IMPORTANT POINTS:
    # For pre-activation Residual Units, make sure to do RELU BEFORE the first path split/addition and AFTER the last element-wise addition.
    assert (depth == 50 or depth == 101), "Depth needs to be either 50 or 101"

    UNIT_STRIDE = [1, 1, 1, 1]
    DOUBLE_STRIDE = [1,2,2,1]
    SAME_STR = 'SAME'
    VALID_STR = 'VALID'
    LONG_REPEAT = 23 if depth == 101 else 6
    LAYER_MULTIPLES = [3, 4, LONG_REPEAT, 3]

    # Units strides with SAME padding results in NO change in feature_map sizes
    def _residual_unit(data_in, f_1x1_1, f_3x3, f_1x1_2, downsampling):
        _, _, _, Fi = data_in.shape
        conv_1x1_1 = bn_relu_conv(data_in, [1, 1, Fi, f_1x1_1], padding=SAME_STR, strides=UNIT_STRIDE, name='conv_1x1_1', training=is_training)
        Fi = conv_1x1_1.shape[3]
        selected_stride = UNIT_STRIDE if not downsampling else DOUBLE_STRIDE
        conv_3x3 = bn_relu_conv(conv_1x1_1, [3, 3, Fi, f_3x3], padding=SAME_STR, strides=selected_stride, name='conv_3x3', training=is_training)
        Fi = conv_3x3.shape[3]
        conv_1x1_2 = bn_relu_conv(conv_3x3, [1, 1, Fi, f_1x1_2], padding=SAME_STR, strides=UNIT_STRIDE, name='conv_1x1_2', training=is_training)

        return conv_1x1_2

    #equation for calculating shape-out is o = [i + 2*pad - kernel]/strides + 1
    #initial input is 300 x 300 x 1

    with tf.variable_scope('conv1'):
        conv1_out = conv_bn_relu(x, kernel_size=[7, 7, 1, 64], strides=DOUBLE_STRIDE, padding=VALID_STR, name='conv1', training=is_training) # 147 x 147 x 64

    max_pool_1 = tf.nn.max_pool(conv1_out, ksize=[1,3,3,1], strides=DOUBLE_STRIDE, name='max_pool_1', padding=SAME_STR) # 74 x 74 x 64
    assert (max_pool_1.shape[1:] == (74, 74, 64)), "Shape of max_pool_1 incorrect."

    with tf.variable_scope('max_pool_project'):
        proj_pool = projection(max_pool_1, 256, down_sample=False)
    assert (proj_pool.shape[1:] == (74, 74, 256)), "Shape of max_pool_project incorrect"

    with tf.variable_scope('conv2_1'):
        conv2_1 = _residual_unit(max_pool_1, 74, 74, 256, downsampling=False)
        prev_output = _add_units(proj_pool, conv2_1)

    #First multi-residual unit layer
    for i in range(LAYER_MULTIPLES[0]-1):
        with tf.variable_scope('conv2_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 64, 64, 256, downsampling=False)
            prev_output = _add_units(out_now, prev_output)
    assert (prev_output.shape[1:] == (74, 74, 256)), "Shape after conv2 incorrect"

    # First layer of conv3 with downsampling
    # First projection since feature_map sizes are now going to be different.
    # conv2 sizes is 74 x 74 x 256.  conv3 sizes are 37 x 37 x 512
    with tf.variable_scope('conv3_1'):
        conv3_1 = _residual_unit(prev_output, 128, 128, 512, downsampling=True) # shape is 37 x 112 x 512
        proj_2_3 = projection(prev_output, 512) #should be size 37 x 37 x 512
        prev_output = _add_units(proj_2_3, conv3_1)

    #Second multi-residual unit layer from conv3_2 on
    for i in range(LAYER_MULTIPLES[1]-1):
        with tf.variable_scope('conv3_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 128, 128, 512, downsampling=False)
            prev_output = _add_units(out_now, prev_output)
    assert (prev_output.shape[1:] == (37, 37, 512)), "Shape after conv3 incorrect"

    #2nd projection level
    with tf.variable_scope('conv4_1'):
        conv4_1 = _residual_unit(prev_output, 256, 256, 1024, downsampling=True) #size 19 x 19 x 1024
        proj_3_4 = projection(prev_output, 1024) # size 19 x 19 x 1024
        prev_output = _add_units(conv4_1, proj_3_4)

    aux_out = auxillary_classifier(prev_output, 700, fc_2=n_classes, is_training=is_training, weight_decay=weight_decay)

    print('3rd multi-layer depth: {}'.format(LAYER_MULTIPLES[2] -1))
    # Third multi-residual unit layer
    for i in range(LAYER_MULTIPLES[2] -1):
        with tf.variable_scope('conv4_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 256, 256, 1024, downsampling=False)
            prev_output = _add_units(out_now, prev_output) # size of 19 x 19 x 1024
    assert (prev_output.shape[1:] == (19, 19, 1024)), "Shape after conv4 incorrect"

    #3rd projection level
    with tf.variable_scope('conv5_1'):
        conv5_1 = _residual_unit(prev_output, 512, 512, 2048, downsampling=True)
        proj_4_5 = projection(prev_output, 2048)
        prev_output = _add_units(conv5_1, proj_4_5) # output shape of 10 x 10 x 2048

    # Fourth/last multi-residual unit layer
    for i in range(LAYER_MULTIPLES[3] -1):
        with tf.variable_scope('conv5_{}'.format(i+2)):
            out_now = _residual_unit(prev_output, 512, 512, 2048, downsampling=False)
            prev_output = _add_units(out_now, prev_output) # size of 10 x 10 x 2048
    assert (prev_output.shape[1:] == (10, 10, 2048)), "Shape after conv5 incorrect"


    with tf.variable_scope('final_output'):
        post_relu = tf.nn.relu(prev_output, name='post_relu')
        _avg_pool = tf.nn.avg_pool(post_relu, ksize=[1, 10, 10, 1], strides=UNIT_STRIDE, padding=VALID_STR, name='avg_pool') #outsize is now 1 x 1 x 2048
        _dropout = tf.layers.dropout(_avg_pool, rate=keeping_rate, name='dropout', training=is_training)
        N, H, W, F = _dropout.shape
        _flattened = tf.reshape(_avg_pool, [-1, int(H*W*F)], name='flattened')
        N, D = _flattened.shape
        FC_W = init_weight('FC_W', shape=[D, n_classes], alpha=weight_decay)
        FC_b = init_bias('FC_b', shape=n_classes)
        logits = tf.matmul(_flattened, FC_W) + FC_b

        return logits, aux_out


# When input images are size 300 x 300 x 1
def Xception_with_auxiliary(x, is_training, n_classes, weight_decay, keeping_rate=0.5):

    UNIT_STRIDE = [1, 1, 1, 1]
    DOUBLE_STRIDE = [1,2,2,1]
    SAME_STR = 'SAME'
    VALID_STR = 'VALID'
    LAYERS_NUM = 8

    aux_out = None

    curr_layer = conv_bn_relu(x, [3, 3, 1, 32], strides=DOUBLE_STRIDE, padding=SAME_STR, name='conv1', training=is_training)
    curr_layer = conv_bn_relu(curr_layer, [3, 3, 32, 64], strides=UNIT_STRIDE, padding=SAME_STR, name='conv2', training=is_training)

    with tf.variable_scope('double_1'):
        curr_layer = double_separable_pool_and_add(curr_layer, 128, 128, True, is_training)

    with tf.variable_scope('double_2'):
        curr_layer = double_separable_pool_and_add(curr_layer, 256, 256, False, is_training)

    with tf.variable_scope('double_3'):
        curr_layer = double_separable_pool_and_add(curr_layer, 728, 728, False, is_training)

    for i in range(LAYERS_NUM):
        with tf.variable_scope('triple_{}'.format(i+1)):
            curr_layer = triple_separable_and_add(curr_layer, 728, training=is_training)

        if i == 3:
           aux_out = auxillary_classifier(curr_layer, 700, fc_2=n_classes, is_training=is_training, weight_decay=weight_decay)


    with tf.variable_scope('double_exit'):
        curr_layer = double_separable_pool_and_add(curr_layer, 728, 1024, False, is_training)

    with tf.variable_scope('conv3'):
        curr_layer = simple_separable_relu(curr_layer, 1536, training=is_training)

    with tf.variable_scope('conv4'):
        curr_layer = simple_separable_relu(curr_layer, 2048, training=is_training)

    with tf.variable_scope('final_output'):
        curr_layer = tf.nn.avg_pool(curr_layer, [1, 10, 10, 1], strides=UNIT_STRIDE, padding=VALID_STR, name='global_avg')
        curr_layer = tf.layers.dropout(curr_layer, rate=keeping_rate, name='dropout', training=is_training)
        N, H, W, F = curr_layer.shape
        _flattened = tf.reshape(curr_layer, [-1, int(H*W*F)], name='flattened')
        N, D = _flattened.shape
        FC_W = init_weight('FC_W', shape=[D, n_classes], alpha=weight_decay)
        FC_b = init_bias('FC_b', shape=n_classes)
        logits = tf.matmul(_flattened, FC_W) + FC_b

        return logits, aux_out

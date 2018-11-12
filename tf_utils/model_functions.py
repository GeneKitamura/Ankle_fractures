import tensorflow as tf
import numpy as np

UNIT_STRIDE = [1, 1, 1, 1]
DOUBLE_STRIDE = [1,2,2,1]
THREE_POOL = [1, 3, 3, 1]
SAME_STR = 'SAME'
VALID_STR = 'VALID'

def init_weight(name, shape, alpha=None, initialize= tf.truncated_normal_initializer(stddev=0.1), should_train=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initialize, dtype=np.float32, trainable=should_train)
        if alpha is not None: #L2 regularization
            L2_loss = tf.multiply(tf.nn.l2_loss(var), alpha, name='L2_loss')
            tf.add_to_collection('l2_loss', L2_loss)
    return var

def init_bias(name, shape, q_val=0.1):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.constant_initializer(q_val), dtype=np.float32)
    return var

def bn_relu_conv(data_in, kernel_size, strides, padding, name, training):
    bn = tf.layers.batch_normalization(data_in, axis=-1, training=training, name = (name+'/bn'), fused=True)
    activation = tf.nn.relu(bn, name=(name+'/relu'))
    kernel = init_weight(name =(name + '/kernel'), shape=kernel_size)
    conv = tf.nn.conv2d(activation, kernel, strides=strides, padding=padding, name=(name + '/conv'))
    bias = init_bias(name = (name + '/bias'), shape=kernel_size[-1])

    return tf.nn.bias_add(conv, bias, name=(name + '/out'))

# Projection function for Resnet
def projection(data_in, out_size, down_sample=True):
    Fi = data_in.shape[3]
    kernel = init_weight('project_kernel', [1, 1, Fi, out_size])

    return tf.nn.conv2d(data_in, kernel, strides=DOUBLE_STRIDE if down_sample else UNIT_STRIDE, padding='SAME', name='projection')

def conv_bn_relu(data_in, kernel_size, strides, padding, name, training): #just like the TF website Variables section
    kernel = init_weight(name = (name + '/kernel'), shape=kernel_size)
    # for tf.nn.conv2d, kernel is [HH, WW, C, F] and strides is [1, stride, stride, 1].  Input is [batch, H, W, C]. Output is [batch, H, W, C]
    conv = tf.nn.conv2d(data_in, kernel, strides, padding=padding)
    biases = init_bias(name = (name + '/bias'), shape=kernel_size[-1])
    pre_bn = tf.nn.bias_add(conv, biases)
    pre_activation = tf.layers.batch_normalization(pre_bn, axis=-1, training=training, name=(name + '/bn'), fused=True)

    return tf.nn.relu(pre_activation, name=(name + '/relu')) #batch x H x W x C

def _add_units(inp_1, inp_2):
    return tf.add_n([inp_1, inp_2], name='add_units')

def double_separable_pool_and_add(inp_x, intermediate_features, out_features, first, training):
    orig_inp = inp_x

    if not first:
        inp_x = tf.nn.relu(inp_x, name='first_relu')

    in_features = inp_x.shape[3]
    kernel1_depth = init_weight('kernel1_depth', [3, 3, in_features, 1])
    kernel1_point = init_weight('kernel1_point', [1, 1, in_features, intermediate_features])
    inp_x = tf.nn.separable_conv2d(inp_x, kernel1_depth, kernel1_point, strides=UNIT_STRIDE, padding=SAME_STR, name='separable_1')
    bias_1 = init_bias('bias_1', shape=inp_x.shape[-1])
    inp_x = tf.nn.bias_add(inp_x, bias_1)
    inp_x = tf.layers.batch_normalization(inp_x, axis=-1, training=training, name='bn_1', fused=True)
    inp_x = tf.nn.relu(inp_x, 'relu_1')

    kernel2_depth = init_weight('kernel2_depth', [3, 3, intermediate_features, 1])
    kernel2_point = init_weight('kernel2_point', [1, 1, intermediate_features, out_features])
    inp_x = tf.nn.separable_conv2d(inp_x, kernel2_depth, kernel2_point, strides=UNIT_STRIDE, padding=SAME_STR, name='separable_2')
    bias_2 = init_bias('bias_2', shape=inp_x.shape[-1])
    inp_x = tf.nn.bias_add(inp_x, bias_2)
    inp_x = tf.layers.batch_normalization(inp_x, axis=-1, training=training, name='bn_2', fused=True)
    inp_x = tf.nn.max_pool(inp_x, ksize=THREE_POOL, strides=DOUBLE_STRIDE, name='max_pool', padding=SAME_STR)

    orig_inp = projection(orig_inp, out_features, down_sample=True)

    return _add_units(inp_x, orig_inp)

def triple_separable_and_add(inp_x, out_features, training):

    orig_inp = inp_x

    for i in range(3):
        with tf.variable_scope('midflow_sub_{}'.format(i)):
            inp_x = tf.nn.relu(inp_x, 'relu')
            in_features = inp_x.shape[-1]
            kernel_depth = init_weight('kernel_depth', [3, 3, in_features, 1])
            kernel_point = init_weight('kernel_point', [1, 1, in_features, out_features])
            inp_x = tf.nn.separable_conv2d(inp_x, kernel_depth, kernel_point, strides=UNIT_STRIDE, padding=SAME_STR, name='separable_triple')
            bias = init_bias('bias', shape=inp_x.shape[-1])
            inp_x = tf.nn.bias_add(inp_x, bias)
            inp_x = tf.layers.batch_normalization(inp_x, axis=-1, training=training, name='bn_2', fused=True)

    return _add_units(inp_x, orig_inp)

def simple_separable_relu(inp_x, out_features, training):
    in_features = inp_x.shape[-1]
    kernel_depth = init_weight('kernel_depth', [3, 3, in_features, 1])
    kernel_point = init_weight('kernel_point', [1, 1, in_features, out_features])
    inp_x = tf.nn.separable_conv2d(inp_x, kernel_depth, kernel_point, strides=UNIT_STRIDE, padding=SAME_STR, name='separable_single')
    bias = init_bias('bias', shape=inp_x.shape[-1])
    inp_x = tf.nn.bias_add(inp_x, bias)
    inp_x = tf.layers.batch_normalization(inp_x, axis=-1, training=training, name='bn', fused=True)

    return inp_x

def auxillary_classifier(data_in, fc_1, fc_2, is_training, weight_decay):
    # for inceptino v3, the input was 17 x 17
    # fc_1 and fc_2 refers to the size of the FC output.

    with tf.variable_scope('auxillary_classifier'):
        _avg_pool = tf.nn.avg_pool(data_in, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID', name='_avg_pool') #for inception v3, output now is 5x5
        _, _, _, F = _avg_pool.shape
        _conv_1x1 = conv_bn_relu(_avg_pool, kernel_size=[1, 1, F, 128], strides=[1, 1, 1, 1], padding='SAME', name='_conv_1x1', training=is_training)

        N, H, W, F = _conv_1x1.shape
        conv_flattened = tf.reshape(_conv_1x1, [-1, int(H*W*F)], name='conv_flattened')
        N, D = conv_flattened.shape
        FC_W1 = init_weight('FC_W1', shape=[D, fc_1], alpha=weight_decay) #fc_1 is hyperparameter; pick maybe ~700?
        FC_b1 = init_bias('FC_b1', shape=fc_1)
        FC_1_NN_relu = tf.nn.relu(tf.matmul(conv_flattened, FC_W1) + FC_b1, name='FC_1_NN_relu')

        FC_W2 = init_weight('FC_W2', shape=[fc_1, fc_2], alpha=weight_decay) # fc_2 is the final output class number; for binary cases, should be 2
        FC_b2 = init_bias('FC_b2', shape=[fc_2])
        FC_2_NN = tf.matmul(FC_1_NN_relu, FC_W2) + FC_b2 #size is N x 2

        return FC_2_NN # before the softmax; just the logits.  The losses need to be weighted; in paper, is was weighted by 0.3.

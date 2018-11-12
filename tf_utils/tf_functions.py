import tensorflow as tf
from sklearn.utils import shuffle

#----------------------Mapping functions---------------------------
# Works for tf.Datasets with ELEMENTS of (([views,299,299,1]), ([])), where each element has the same STRUCTURE.  The first component of the Dataset element is x with shape of (views, 299, 299, 1) and the second component is y with shape of ().
# Performs the designated function on each element.  If batch of data, then performs the designated (mapped) function to each sample of the batch.

# for test and validation sets WITHOUT data augmentation
def view_merging(d_input, y_input, multi_views=None, horizontal_merge=True):
    merge_axis = 2 if horizontal_merge else 1 # else results in vertical merge
    if multi_views:
        d_input = tf.squeeze(tf.concat(tf.split(d_input, multi_views, axis=0), axis=merge_axis), axis=0) # d_input is now shape 299 x 897 x 1 or something similar if horizontal merge
    else:
        pass  # if not multi_views, then just (299, 299 x 1) or something like that.
    d_input = tf.image.per_image_standardization(d_input)
    return d_input, y_input

# input of inp_x is shape multi_views x 299 x 299 x 1 as each component of the 1st element
# or input could be just a single view of (299 x 299 x 1)
# input of inp_y is shape None as each component of the 2nd element
def data_augmentation(inp_x, inp_y, multi_views=None, horizontal_merge =True, bright_con=0.3, low_contrast=0.7, high_contrast=1.3):
    if multi_views:
        inp_x = shuffle(tf.split(inp_x, multi_views, axis=0)) # end up with list of 3 x (1 x 299 x 299 x 1) with shuffling
    else:
        inp_x = [tf.expand_dims(inp_x, axis=0)] # wrap the single view in a list for the upcoming enumerate

    # can ONLY enumerate through a list since a tensor object is NOT iterable
    for i, j in enumerate(inp_x):
        tmp = tf.image.random_brightness(tf.squeeze(j, axis=0), bright_con) # get rid of axis=0 with squeeze
        tmp = tf.image.random_contrast(tmp, low_contrast, high_contrast)
        tmp = tf.image.random_flip_up_down(tmp)
        tmp = tf.image.random_flip_left_right(tmp)
        inp_x[i] = tmp # inp_x is list of 3 and elements shaped 299 x 299 x 1

    merge_axis = 1 if horizontal_merge else 0
    inp_x = tf.concat(inp_x, axis=merge_axis) # now is single tensor shape 299 x 897 x 1 if horizontal merge, otherwise, just 299 x 299 x 1 or 300 x 300 x 1
    inp_x = tf.image.per_image_standardization(inp_x)

    return inp_x, inp_y

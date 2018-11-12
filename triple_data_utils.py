import numpy as np
import pydicom as dicom
import os
import matplotlib.pyplot as plt
from glob import glob
import re
from math import floor
from skimage import transform, exposure, img_as_float
import skimage
from scipy import misc
from sklearn.utils import shuffle
import tensorflow as tf

both_dir = '~/PycharmProjects/Machine/CNN/large_files'
NL_dir = 'all_NL'
ABNL_dir = 'all_ABNL'

#-------------------------MULTI VIEWS READING----------------------------
# for parsing folders of patients into a list of patients with a MAX of 3 of their views
def _folder_parse(curr_dir, view_expand=False, add_grayscale_channel=True, crop=True):
    curr_three = []
    curr_error = []
    curr_non3 = []
    curr_one = []
    curr_two = []

    for folder in os.listdir(curr_dir):
        curr_case = []

        for j in os.listdir(os.path.join(curr_dir, folder)):
            final_file_path = os.path.join(curr_dir, folder, j)
            try:
                raw_dcm = dicom.read_file(final_file_path) #returns a numpy array
                assert(raw_dcm.pixel_array.dtype == 'uint16'), "The dicom int type is NOT uint16"

                tmp = exposure.rescale_intensity(raw_dcm.pixel_array, in_range=('uint' + str(raw_dcm.BitsStored)))
                if crop:
                    tmp = _crop_data(tmp) # get rid of DEFINITELY non-relevant pixels
                tmp = np.float32(img_as_float(tmp))
                if add_grayscale_channel:
                    tmp = np.expand_dims(tmp, axis=2) #adding grayscale channel
                curr_case.append(tmp)

            except (IsADirectoryError, TypeError):
                curr_error.append(final_file_path)

            if len(curr_case) == 3:
                break

        if len(curr_case) == 0: # no valid cases, it will be skipped
            continue

        if len(curr_case) == 1:
            curr_one.append(curr_case)

        if len(curr_case) == 2:
            curr_two.append(curr_case)

        #keep track of cases with less than 3 views:
        if len(curr_case) < 3:
            if view_expand:
                while len(curr_case) < 3: #copy a random view from list of curr_case
                    curr_case.append(curr_case[np.random.randint(len(curr_case))])
                curr_non3.append(curr_case)

            else:
                continue
        else:
            curr_three.append(curr_case)

    return curr_three, curr_two, curr_one

# returns the cases and y as list.  the images are float representation with type of np.float32.
def _concatenated_check_and_labels(base_dir, NL_sub, ABNL_sub, expand=False, return_non3=True, grayscale_channeling=True):
    NL_dir = os.path.join(base_dir, NL_sub) #directory containing all NL folder
    ABNL_dir = os.path.join(base_dir, ABNL_sub) #directory containing all ABNL folder

    NL_three, NL_two, NL_one = _folder_parse(NL_dir, expand, grayscale_channeling)
    ABNL_three, ABNL_two, ABNL_one = _folder_parse(ABNL_dir, expand, grayscale_channeling)

    # Everything is returned as a list
    if return_non3:
        return (NL_three, NL_two, NL_one), (ABNL_three, ABNL_two, ABNL_one)
    else:
        return NL_three, ABNL_three

# Take patient list with sublist of multiple views, and merged it into one list
def _unpack_views(pt_list):
    tot_list = []
    for view_list in pt_list:
        for view in view_list:
            tot_list.append(view)
    return tot_list

def _single_views_resize(single_views_list, final_size=(300, 300, 1)):
    return np.array([transform.resize(i, output_shape=final_size, order=3, mode='reflect') for i in single_views_list], np.float32)

# need to work with lists because numpy arrays have a designated shape, while list are designated only by its index
# Will take in 3 views per patient and resize each view
def _resize_after_concat(pt_3view_list, final_size=(300, 300, 1), range_preserved=False):

    #resize the images
    for i, j in enumerate(pt_3view_list):
        pt_3view_list[i] = [transform.resize(k, output_shape=final_size, order=3, mode='reflect', preserve_range=range_preserved) for k in j]

    # CAUTION - original list is being modified
    return np.array(pt_3view_list, np.float32)


#----------------CROPPING IMAGES ------------------------
# takes in a list of images
def _crop_image_list(image_list):
    return [_crop_data(i) for i in image_list]

# takes in a single image numpy array
def _crop_data(single_image):
    # get rid of unnecessary borders if the pixel values are equal along the rows or columns

    current_image = single_image

    # checking the rows from the top
    for i in range (current_image.shape[0]):
        pixel_check = current_image[i][0]

        # Need to break out of loop when pixels are no longer homogeneous across row
        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[i:]
            break

    # checking rows from the bottom
    for i in range(current_image.shape[0]-1, 0, -1): # minus one at starting point to prevent IndexError
        pixel_check = current_image[i][0]

        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[:i]
            break

    # checking columns from one side
    for i in range(current_image.shape[1]):
        pixel_check = current_image[0][i]

        # Need to break out of loop when pixels are not longer homogeneous through column
        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, i:]
            break

    # checking columns from the other side
    for i in range(current_image.shape[1] - 1, 0, -1):
        pixel_check = current_image[0][i]

        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, :i]
            break

    return current_image


#-----------------------STANDARDIZE-----------------------------
def _standardize(resized_list, axis_standardize=True):
    if axis_standardize:
        resized_list = (resized_list - np.mean(resized_list, axis=0))/(np.std(resized_list, axis=0) + 1e-5)
    else:
        resized_list = (resized_list - np.mean(resized_list))/(np.std(resized_list) + 1e-5)

    return resized_list


#------------------- Manual data-augementation ----------------------------------
# Must have tf.Interactive session instantiated for tf.image methods to be eval().
# performs random brightness, contrast, flip-LR, and flip-UD on each image for num_iters
def data_augmentation(image_array, num_iters=5, alt_bright=30, lower_cont=0.7, upper_cont=1.3):
    aug_list = []
    for three_views in image_array:
        for i in range(num_iters):
            changed_views = []
            for one_image in shuffle(three_views):
                one_image = tf.image.random_brightness(one_image, alt_bright)
                one_image = tf.image.random_contrast(one_image, lower_cont, upper_cont)
                one_image = tf.image.random_flip_left_right(one_image)
                one_image = tf.image.random_flip_up_down(one_image)
                changed_views.append(one_image.eval())
            aug_list.append(changed_views)

    return np.array(aug_list, np.float32)

# Merge the 3 views into 1 numpy array and reshape into N x H X W x C
# change com_axis to 1 for vertical merge
# i has 4 dimension (3 x H x W x C), while image array has 5 dimensions (N x 3 X H x W x C)


#---------------------------3 views list into 1 numpy array --------------------------
def merge_3_views(image_array, com_axis=2, del_axis=1):
    return np.squeeze(np.array([np.concatenate(np.split(i, 3, axis=0), axis=com_axis) for i in image_array]), axis=del_axis)



#------------------NON-UNIFORM SIZE DATA----------------------------

# take in a list of image with NON-UNIFORM SHAPES and calculate mean and std dev across all pixels as a SINGLE mean and std-dev
# to be used before padding the images.
# NOT needed when the individual images are already resized.
def _standardization_params(image_list):
    total = 0
    count = 0
    N = len(image_list)
    for i in range(N):
        total += np.sum(image_list[i])
        count += np.prod((image_list[i].shape[0], image_list[i].shape[1]))
    image_average = total/count

    squared_diff = 0
    for i in range(N):
        squared_diff += np.sum((image_list[i] - image_average)**2)
    image_std = np.sqrt(squared_diff/count)

    return image_average, image_std

# standardizes all images in a NON-UNIFORM SHAPED list by subtracting a single mean and dividing by single std dev.
def _images_standardization(image_list):
    image_average, image_std = _standardization_params(image_list)
    return (image_list - image_average) / (image_std + 1e-5) #with smoothing


# ---------------------CHECKING SIZES-----------------------
# takes in array of cleaned images
def _eval_sizes(cleaned_image_list):
    max_width = []
    max_height = []
    for i in cleaned_image_list:
        max_width.append(i.shape[0])
        max_height.append(i.shape[1])
    return max_width, max_height


# takes in pt dictionary, such as from _order_data
def _check_sizes(total_pt_dict):
    row_sizes = []
    column_sizes = []
    for key, value in total_pt_dict.items():
        for i in range(len(value)):
            row_sizes.append(value[i].shape[0])
            column_sizes.append(value[i].shape[1])

    max_row_size = max(row_sizes)
    max_column_size = max(column_sizes)

    return max_row_size, max_column_size

#--------------------------SHOWING IMAGES-----------------------
# takes in a pt dictionary
def _show_dictionary_images(total_pt_dict):
    for key, value in total_pt_dict.items():
        for i in range(len(value)):
            plt.imshow(value[i], cmap=plt.cm.gray)
            plt.title(key)
            plt.show()

# takes in a pt array
def _show_array_of_images(patient_array, label_array, channel=True, num_row=10, num_col=10, curr_min=None, curr_max=None):
    f, ax = plt.subplots(num_row, num_col, figsize=(10, 10))
    f.subplots_adjust(hspace=None, wspace=None)

    # Work-around for single image
    if type(ax) is not np.ndarray:
        ax = np.array([ax], dtype=np.dtype)
        patient_array = np.expand_dims(patient_array, axis=0)

    for i, j in zip(range(len(patient_array)), ax.flatten()):
        if channel:
            j.imshow(patient_array[i][...,0], cmap=plt.cm.gray, vmin=curr_min, vmax=curr_max)
        else:
            j.imshow(patient_array[i], cmap=plt.cm.gray, vmin=curr_min, vmax=curr_max)
        j.axis('off')
        if label_array is not None:
            j.set_title("{}".format(label_array[i]))


# for just 3 views
def _three_view_show(patient_list, titling, curr_min=None, curr_max=None):
    f, ax = plt.subplots(1, 3, figsize=(20, 20))
    f.subplots_adjust(hspace=0, wspace=0)

    for i, j in zip(range(len(patient_list)), ax.flatten()):
        j.imshow(patient_list[i][...,0], cmap=plt.cm.gray, vmin=curr_min, vmax=curr_max)
        j.set_title('Broken' if titling == 1 else 'Normal')
        j.axis('off')


def overlapping_heat_map(images, labels, model, patch_sz=100, hor_loc=5, ver_loc=5, save='False'):
    images_N, images_H, images_W, input_channels = images.shape
    assert (patch_sz*hor_loc >= images_W), "Not enough horizontal locations/data points, increase hor_loc"
    assert (patch_sz*ver_loc >= images_H), "Not enough vertical locations/data points; increase ver_loc"
    assert (hor_loc > 1), "hor_loc must be greater than 1"
    assert (ver_loc > 1), "ver_loc must be greater than 1"

    zeroed_patch = np.zeros((patch_sz, patch_sz, input_channels))
    output_heat_map = np.zeros((images_H, images_W))

    # to keep count of how many times each pixel was overlapped;
    count_heat_accuracy = np.zeros((images_H, images_W)) + 1e-5 #smoothing

    # 1st, find the last POSSIBLE position of patching:
    last_ver_pos = images_H - patch_sz
    last_hor_pos = images_W - patch_sz

    # Next, divide the intervening pixel positions by the number of specified locations:
    hor_pos = last_hor_pos/(hor_loc-1)
    ver_pos = last_ver_pos/(ver_loc- 1) # minus one since last position is accounted for above
    count = 0

    for i in range(ver_loc):

        for j in range(hor_loc):
            zeroed_images = np.zeros_like(images)
            # Making a copy of the original images
            zeroed_images[...] = images[...]

            zeroed_images[:,round(i*ver_pos): round((i*ver_pos+patch_sz)), round(j*hor_pos): round(j*hor_pos+patch_sz), :] = zeroed_patch

            print("Position {} of {}".format((count), ver_loc*hor_loc))
            count += 1
            curr_accuracy = model.test_model(zeroed_images, labels, True, False, False)
            output_heat_map[round(i*ver_pos): round((i*ver_pos+patch_sz)), round(j*hor_pos): round(j*hor_pos+patch_sz)] += curr_accuracy
            count_heat_accuracy[round(i*ver_pos): round((i*ver_pos+patch_sz)), round(j*hor_pos): round(j*hor_pos+patch_sz)] += 1

    averaged_map = output_heat_map / count_heat_accuracy

    if save:
        np.save('./heatmap', averaged_map)

    return averaged_map


def show_heat_map(cur_heat_map, color_scheme='RdYlBu', interp='bicubic', v_min=0.6, v_max=0.8, alpha=1.0, save=False, height=20, width=20, format='eps', dpi=600):
    ground_fig, ground_axes = plt.subplots(figsize=(height, width)) # when axes > 1, the Axes are held in a numpy.darray.  If axes == 1, it's just a plt object.
    plot_heat_map = ground_axes.imshow(cur_heat_map, cmap=color_scheme, interpolation=interp, vmin=v_min, vmax=v_max, alpha=alpha)
    print('Max accuracy: {}'.format(np.max(cur_heat_map.flatten())))
    print('Min accuracy: {}'.format(np.min(cur_heat_map.flatten())))
    plt.colorbar(plot_heat_map)

    if save:
        plt.savefig('./plots/heat_map', format=format, dpi=dpi)

    plt.show()

def superimposed_images(image, segmentation, color_scheme='hot', alpha=0.5, save=False, height=20, width=20, format='eps', dpi=600):
    f, ax = plt.subplots(1, 1, figsize=(height, width))

    ax.imshow(image, cmap='gray')
    ax.imshow(segmentation, cmap=color_scheme, alpha=alpha)
    ax.axis('off')

    if save:
        plt.savefig('./plots/superimposed_map', format=format, dpi=dpi)

    plt.show()

# H x W x C images
def tandem_view(image1, image2, save=False, height=20, width=20, format='eps', dpi=600, name='some_output'):

    fig, axes = plt.subplots(1, 2, subplot_kw= {}, figsize=(height, width))
    fig.subplots_adjust(hspace=0, wspace=0)

    axes[0].imshow(image1[..., 0], cmap='gray')
    axes[0].axis('off')
    axes[0].text(0.1, 0.9, 'Fig.2a', transform=axes[0].transAxes, color='white', verticalalignment='bottom', horizontalalignment='left', fontsize=8)

    axes[1].imshow(image2[..., 0], cmap='gray')
    axes[1].axis('off')
    axes[1].text(0.1, 0.9, 'Fig.2b', transform=axes[1].transAxes, color='white', verticalalignment='bottom', horizontalalignment='left', fontsize=8)

    if save:
        plt.savefig('./plots/{}'.format(name), format=format, dpi=dpi)

    plt.show()

def filter_heat_map_tandem_view(orig_img, curr_heat_map, heat_color='seismic_r', accuracy_cut_off=0.71, save=False, height=10, width=10, format='eps', dpi=600, name='tandem_map'):

    fig, axes = plt.subplots(1, 2, subplot_kw= {}, figsize=(height, width))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Get rid of channel
    orig_img = orig_img[..., 0]

    axes[0].imshow(orig_img, cmap='gray')
    axes[0].imshow(curr_heat_map, cmap=heat_color, alpha=0.5)
    axes[0].axis('off')
    axes[0].text(0.1, 0.8, 'Fig.3a', transform=axes[0].transAxes, color='black', verticalalignment='bottom', horizontalalignment='left', fontsize=8)

    # Creating filter/mask from heatmap based on accuracy cut off
    curr_mask = curr_heat_map < accuracy_cut_off
    filtered_img = curr_mask * orig_img

    axes[1].imshow(filtered_img , cmap='gray')
    axes[1].axis('off')
    axes[1].text(0.1, 0.8, 'Fig.3b', transform=axes[1].transAxes, color='white', verticalalignment='bottom', horizontalalignment='left', fontsize=8)

    if save:
        plt.savefig('./plots/{}'.format(name), format=format, dpi=dpi)

    plt.show()
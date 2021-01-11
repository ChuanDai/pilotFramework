from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def get_frames_data(filename, num_frames_per_clip=16):
    """
    Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays.

    :param filename: A directory that contains set of frame images.
    :param num_frames_per_clip: The number of consecutive frames.

    :return ret_arr: Saves a list of frame data.
    :return s_index: Start frame index.
    """
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames)<num_frames_per_clip:
            return [], s_index
        filenames = sorted(filenames)
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, s_index


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
    """
    Obtain and preprocess the data from one batch in the dataset.

    :param filename: The List file that holds the data set information.
    :param batch_size: The sample size of a batch.
    :param start_pos: The starting sample index for the batch.
    :param num_frames_per_clip: The number of consecutive frames.
    :param crop_size: The size of each frame after the image is cropped.
    :param shuffle: Whether to scramble the data set order.

    :return np_arr_data: A NumPy array for saving data of a batch.
    :return np_arr_label: The NumPy array that holds a batch of data for the corresponding label.
    :return next_batch_start: The starting sample index for the next batch.
    :return read_dirnames: Save a list of the corresponding folder paths of the batch data.
    :return valid_len: The number of valid (non-duplicate) samples in the batch.
    """
    lines = open(filename, 'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    # forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        # https://stackoverflow.com/questions/20484195/typeerror-range-object-does-not-support-item-assignment
        # video_indices = range(len(lines))
        video_indices = list(range(len(lines)))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if batch_index >= batch_size:
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
        tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
        img_datas = []
        if len(tmp_data) != 0:
            for j in xrange(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))
                if img.width > img.height:
                    scale = float(crop_size)/float(img.height)
                    img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
                else:
                    scale = float(crop_size)/float(img.width)
                    img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
                crop_x = int((img.shape[0] - crop_size)/2)
                crop_y = int((img.shape[1] - crop_size)/2)
                img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :] - np_mean[j]
                img_datas.append(img)
            data.append(img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

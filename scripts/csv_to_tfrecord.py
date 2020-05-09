#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# This has to be in the same order as in labels_map.pbtxt
# TODO, read the file -------------------^
LABELS = [
    "stop_sign"
]
CSV_TEST_FILE = "annotations/test_labels.csv"
CSV_TRAIN_FILE = "annotations/train_labels.csv"
RECORD_TEST_FILE = "annotations/test.record"
RECORD_TRAIN_FILE = "annotations/train.record"
IMAGES_TEST_DIR = "images/test"
IMAGES_TRAIN_DIR = "images/train"

def class_text_to_int(row_label):
    index = 1
    for label in LABELS:
        if label == row_label:
            return index
        index = index + 1

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(RECORD_TEST_FILE)
    examples = pd.read_csv(CSV_TEST_FILE)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, IMAGES_TEST_DIR)
        writer.write(tf_example.SerializeToString())
    writer.close()

    writer = tf.io.TFRecordWriter(RECORD_TRAIN_FILE)
    examples = pd.read_csv(CSV_TRAIN_FILE)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, IMAGES_TRAIN_DIR)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.compat.v1.app.run()

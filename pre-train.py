#!/usr/bin/env python3

import os
import io
import re
import sys
import glob
import math
import shutil
import random
import shutil

import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf

sys.path.append("imports")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format

from common import *

def get_labels_map(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


def partition_images(images_dir, images_train_dir, images_test_dir, ratio):
    images = [f for f in os.listdir(images_dir) if re.search(r'^.+\.(jpg|jpeg|png)$', f)]
    images.sort()

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    shutil.rmtree(images_test_dir)
    shutil.rmtree(images_train_dir)

    os.mkdir(images_test_dir)
    os.mkdir(images_train_dir)

    for i in range(num_images):
        filename = images[i]
        filepath_source = os.path.join(images_dir, filename)
        filename_xml = os.path.splitext(filename)[0]+'.xml'
        filepath_source_xml = os.path.join(images_dir, filename_xml)
        filepath_dest = os.path.join(images_train_dir, filename)
        filepath_dest_xml = os.path.join(images_train_dir, filename_xml)

        if i < num_test_images:
            filepath_dest = os.path.join(images_test_dir, filename)
            filepath_dest_xml = os.path.join(images_test_dir, filename_xml)

        shutil.copyfile(filepath_source, filepath_dest)
        shutil.copyfile(filepath_source_xml, filepath_dest_xml)


def xml_to_csv(xml_dir, csv_file):
    xml_list = []
    for xml_file in glob.glob(xml_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(csv_file, index=None)


def class_text_to_int(row_label, labels):
    index = 1
    for label in labels:
        if label == row_label:
            return index
        index = index + 1

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, labels):
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
        classes.append(class_text_to_int(row['class'], labels))

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


def csv_to_tfrecord(record_file, csv_file, images_dir, labels):
    writer = tf.io.TFRecordWriter(record_file)
    examples = pd.read_csv(csv_file)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_dir, labels)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    # Write labels map file
    with open(LABELS_MAP_FILE, 'w') as f:
        txt = get_labels_map(LABELS)
        f.write(txt)

    # Write labels file
    with open(LABELS_FILE, 'w') as f:
        f.writelines("%s\n" % label for label in LABELS)

    # Partition images
    partition_images(IMAGES_DIR, IMAGES_TRAIN_DIR, IMAGES_TEST_DIR, IMAGES_TEST_TO_TRAIN_RATIO)

    # Convert .xml files to .csv
    xml_to_csv(XML_TEST_DIR, CSV_TEST_FILE)
    xml_to_csv(XML_TRAIN_DIR, CSV_TRAIN_FILE)

    # Convert .csv files to .record
    csv_to_tfrecord(RECORD_TEST_FILE, CSV_TEST_FILE, IMAGES_TEST_DIR, LABELS)
    csv_to_tfrecord(RECORD_TRAIN_FILE, CSV_TRAIN_FILE, IMAGES_TRAIN_DIR, LABELS)



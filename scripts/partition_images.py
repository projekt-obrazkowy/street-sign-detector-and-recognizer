import os
import re
import shutil
from PIL import Image
from shutil import copyfile
import argparse
import glob
import math
import random
import xml.etree.ElementTree as ET

SOURCE_DIR = "images"
DEST_DIR = SOURCE_DIR
TEST_TO_TRAIN_RATIO = 0.2

def main():
    source = SOURCE_DIR.replace('\\', '/')
    dest = DEST_DIR.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(TEST_TO_TRAIN_RATIO*num_images)

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(source, filename), os.path.join(test_dir, filename))
        xml_filename = os.path.splitext(filename)[0]+'.xml'
        copyfile(os.path.join(source, xml_filename), os.path.join(test_dir,xml_filename))
        images.remove(images[idx])

    for filename in images:
        copyfile(os.path.join(source, filename), os.path.join(train_dir, filename))
        xml_filename = os.path.splitext(filename)[0]+'.xml'
        copyfile(os.path.join(source, xml_filename), os.path.join(train_dir, xml_filename))

if __name__ == '__main__':
    main()

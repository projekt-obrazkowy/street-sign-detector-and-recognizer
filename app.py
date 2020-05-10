#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('imports')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from common import *

label_map = label_map_util.load_labelmap(LABELS_MAP_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUMBER_OF_MAX_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(INFERENCE_GRAPH_FILE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
session = tf.compat.v1.Session(graph=detection_graph)

for path in sys.argv[1:]:
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num) = session.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=LINE_THICKNESS,
        min_score_thresh=MIN_DETECTION_SCORE
    )

    cv2.imshow(path, image)
    cv2.waitKey()

cv2.destroyAllWindows()

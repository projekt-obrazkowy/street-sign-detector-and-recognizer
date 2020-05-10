LABELS = [
    "stop_sign"
]

NUMBER_OF_MAX_CLASSES = len(LABELS)

LINE_THICKNESS = 6
MIN_DETECTION_SCORE = 0.8
IMAGES_TEST_TO_TRAIN_RATIO = 0.2

LABELS_MAP_FILE = "annotations/label_map.pbtxt"
LABELS_FILE = "annotations/labels.txt"

CSV_TEST_FILE = "annotations/test_labels.csv"
CSV_TRAIN_FILE = "annotations/train_labels.csv"

RECORD_TEST_FILE = "annotations/test.record"
RECORD_TRAIN_FILE = "annotations/train.record"

CSV_TEST_FILE = "annotations/test_labels.csv"
CSV_TRAIN_FILE = "annotations/train_labels.csv"

IMAGES_DIR = "images"
IMAGES_TEST_DIR = "images/test"
IMAGES_TRAIN_DIR = "images/train"

XML_TEST_DIR = "images/test"
XML_TRAIN_DIR = "images/train"

MODEL_DIR = "models/train"

CONFIG_FILE = "configs/ssd_inception_v2_coco.config"

OUTPUT_DIR = "models/post-train"

INFERENCE_GRAPH_FILE = 'models/post-train/frozen_inference_graph.pb'

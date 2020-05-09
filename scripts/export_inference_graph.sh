#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD/imports/slim:$PWD/imports/object_detection"

python3 imports/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/train/ssd_inception_v2_coco.config \
    --trained_checkpoint_prefix models/train/model.ckpt-425 \
    --output_directory models/post-train

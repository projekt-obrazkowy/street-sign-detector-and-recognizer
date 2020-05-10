#!/usr/bin/env python3

import os
import glob
import shutil
import sys
import subprocess

SCRIPT = "imports/object_detection/export_inference_graph.py"
MODEL_DIR = "models/train"
OUTPUT_DIR = "models/post-train"
CONFIG_FILE = "configs/ssd_inception_v2_coco.config"

if __name__ == '__main__':
    indexes = glob.glob(f"{MODEL_DIR}/model.ckpt-*.index")
    ckpt = indexes[0].replace(".index", "")

    shutil.rmtree(OUTPUT_DIR)

    os.environ["PYTHONPATH"] = f"imports:imports/slim"
    r = subprocess.run(["python3", SCRIPT, "--input_type", "image_tensor", "--pipeline_config_path", CONFIG_FILE, "--trained_checkpoint_prefix", ckpt, "--output_directory", OUTPUT_DIR])
    if r.returncode != 0:
        sys.exit(1)

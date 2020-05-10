#!/usr/bin/env python3

import os
import glob
import shutil
import sys
import subprocess

from common import *

SCRIPT = "imports/object_detection/export_inference_graph.py"

if __name__ == '__main__':
    indexes = glob.glob(f"{MODEL_DIR}/model.ckpt-*.index")
    ckpt = indexes[0].replace(".index", "")

    shutil.rmtree(OUTPUT_DIR)

    os.environ["PYTHONPATH"] = f"imports:imports/slim"
    r = subprocess.run(["python3", SCRIPT, "--input_type", "image_tensor", "--pipeline_config_path", CONFIG_FILE, "--trained_checkpoint_prefix", ckpt, "--output_directory", OUTPUT_DIR])
    if r.returncode != 0:
        sys.exit(1)

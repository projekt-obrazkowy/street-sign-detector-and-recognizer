#!/usr/bin/env python3

import os
import sys
import subprocess

SCRIPT = "imports/object_detection/model_main.py"
MODEL_DIR = "models/train"
CONFIG_FILE = "configs/ssd_inception_v2_coco.config"

if __name__ == '__main__':
    os.environ["PYTHONPATH"] = f"imports:imports/slim"
    r = subprocess.run(["python3", SCRIPT, "--model_dir", MODEL_DIR, "--pipeline_config_path", CONFIG_FILE])
    if r.returncode != 0:
        sys.exit(1)

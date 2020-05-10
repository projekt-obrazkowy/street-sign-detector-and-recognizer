#!/usr/bin/env python3

import os
import sys
import subprocess

from common import *

SCRIPT = "imports/object_detection/model_main.py"

if __name__ == '__main__':
    os.environ["PYTHONPATH"] = f"imports:imports/slim"
    r = subprocess.run(["python3", SCRIPT, "--model_dir", MODEL_DIR, "--pipeline_config_path", CONFIG_FILE])
    if r.returncode != 0:
        sys.exit(1)

#!/bin/bash

set -e

python -m pip install .[full] --no-warn-script-location

pyinstaller s3a/__main__.py \
    --clean \
    --noconfirm \
    --log-level=WARN \
    --windowed \
    --name=s3a

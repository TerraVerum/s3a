#!/bin/bash

[[ -d ".wine-0/" ]] && rm -rf .wine-0/

wine pip install -e . --no-dependencies --no-warn-script-location

[[ -d "win_dist" ]] && rm -rf win_dist/

wine pyinstaller s3a/__main__.py \
    --clean \
    --noconfirm \
    --log-level=WARN \
    --windowed \
    --name=s3a \
    --distpath="win_dist"

if [ -n $1 ]; then
    echo "Chowning to give to user and group $1"
    chown -R $1:$1 build/ win_dist/ .wine-0/
    chown $1:$1 s3a.spec
fi

# zip ./win_dist.zip ./win_dist

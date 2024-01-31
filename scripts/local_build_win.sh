#!/bin/bash

docker run --volume "$PWD:/tmp" --workdir "/tmp" tobix/pywine:3.10 ./scripts/build_win.sh $UID

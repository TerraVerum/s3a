#!/bin/bash

set -e

docker build  --tag s3a:local .

docker run --volume "$PWD:/tmp" --workdir "/tmp" s3a:local ./scripts/build_nix.sh $UID

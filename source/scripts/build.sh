#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
cmake --build "${BUILD_FOLDER}" -- -j 4 -v

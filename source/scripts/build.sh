#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
cmake --build "${BUILD_FOLDER}" -- -j 4 -v
if [ "$1" = "release" ]; then
  source "$(dirname "${0}")/misc/objdump/objdump.sh"
fi
echo "Build is ready."

#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
ctest --test-dir ${BUILD_FOLDER}/


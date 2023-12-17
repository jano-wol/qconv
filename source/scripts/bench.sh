#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
${BUILD_FOLDER}/bin/zero-bench
${BUILD_FOLDER}/bin/copy-bench
${BUILD_FOLDER}/bin/dilate-bench
${BUILD_FOLDER}/bin/add-bench
${BUILD_FOLDER}/bin/min_max-bench
${BUILD_FOLDER}/bin/min_max_global-bench
${BUILD_FOLDER}/bin/relu-bench
${BUILD_FOLDER}/bin/linear-bench
${BUILD_FOLDER}/bin/qconv-bench


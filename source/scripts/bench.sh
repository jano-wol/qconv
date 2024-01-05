#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
#${BUILD_FOLDER}/bench/bin/zero-bench
#${BUILD_FOLDER}/bench/bin/copy-bench
#${BUILD_FOLDER}/bench/bin/dilate-bench
#${BUILD_FOLDER}/bench/bin/add-bench
#${BUILD_FOLDER}/bench/bin/min_max-bench
#${BUILD_FOLDER}/bench/bin/min_max_h-bench
#${BUILD_FOLDER}/bench/bin/relu-bench
${BUILD_FOLDER}/bench/bin/linear-bench
${BUILD_FOLDER}/bench/bin/qconv-bench


#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
${BUILD_FOLDER}/test/bin/test_qconv
echo "Qconv tests passed"
echo "All tests passed"


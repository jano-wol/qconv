#!/bin/bash
set -ex
${BUILD_FOLDER}/test/bin/test_loader_eval
echo "test_loader_eval ready"
${BUILD_FOLDER}/test/bin/test_loader_move_candidate
echo "test_loader_move_candidate ready"
echo "load_dummy_data test OK"


#!/bin/bash
set -ex
source ${PYTHON_FOLDER}/venv/bin/activate

output_root_dir="${BUILD_TEST_DATA_FOLDER}/dummy_nnue_move_candidate"
python3 ${PYTHON_FOLDER}/testscript/tester.py ${BUILD_TEST_DATA_FOLDER}/dummy_data_move_candidate.bin move_candidate --default_root_dir ${output_root_dir}
output_root_dir="${BUILD_TEST_DATA_FOLDER}/dummy_nnue_eval"
python3 ${PYTHON_FOLDER}/testscript/tester.py ${BUILD_TEST_DATA_FOLDER}/dummy_data_eval.bin eval --default_root_dir ${output_root_dir}
echo "genereate_dummy_nnue test OK"


#!/bin/bash
set -ex
${BUILD_FOLDER}/test/bin/dummy_data_generator_eval
file=${BUILD_TEST_DATA_FOLDER}/dummy_data_eval.bin
eval256=$(sha256sum ${file} | cut -d' ' -f1)
eval256_expected="e620e46859a17fd3f004a112a34e815d026734a2e8403b5578d5092e6f17105e"
if [[ ${eval256} != ${eval256_expected} ]]; then
    echo -e "sha256 mismatch.\neval256         =${eval256}\neval256_expected=${eval256_expected}"
    exit 1
fi	
echo "dummy_data_generator_eval ready"
${BUILD_FOLDER}/test/bin/dummy_data_generator_move_candidate
file=${BUILD_TEST_DATA_FOLDER}/dummy_data_move_candidate.bin
move256=$(sha256sum ${file} | cut -d' ' -f1)
move256_expected="ed332799b66e323f7537967a84b18be60dcc9c80c679f66a8935d57403b35b67"
if [[ ${move256} != ${move256_expected} ]]; then
    echo -e "sha256 mismatch.\nmove256         =${move256}\nmove256_expected=${move256_expected}"
    exit 1
fi	
echo "dummy_data_generator_move_dandidate ready"
echo "genereate_dummy_data test OK"


#!/bin/bash
set -ex
BIN_FOLDER="${BUILD_FOLDER}/bin"
TARGET_FOLDER="${BUILD_FOLDER}/objdump" 
mkdir -p ${TARGET_FOLDER}
for FILENAME_ABS in "${BIN_FOLDER}"/*; do
   FILENAME=$(basename ${FILENAME_ABS})
   objdump -d "${BIN_FOLDER}/${FILENAME}" > "${TARGET_FOLDER}/objdump_${FILENAME}.txt"
done


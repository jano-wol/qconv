#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
rm -rf "${BUILD_FOLDER}"
mkdir -p "${BUILD_FOLDER}"
cd "${BUILD_FOLDER}"
cmake "${WORKSPACE_FOLDER}" "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}" "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}" -G "Ninja" "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"


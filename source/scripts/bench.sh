#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
${BUILD_FOLDER}/bin/bench


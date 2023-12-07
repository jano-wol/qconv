#!/bin/bash
set -ex
source "$(dirname "${0}")/build/init.sh"
source "$(dirname "${0}")/test/nnue_zero.sh"
source "$(dirname "${0}")/test/generate_dummy_data.sh"
source "$(dirname "${0}")/test/load_dummy_data.sh"
source "$(dirname "${0}")/test/generate_dummy_nnue.sh"
source "$(dirname "${0}")/test/nnue_load.sh"
echo "All tests passed"


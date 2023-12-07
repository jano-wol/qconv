#include <layers/tile.h>

int conv_global[7][TSIZE][15 * 15 + 1];
int conv_rel_global[7][TSIZE][15 * 15 + 1];

void qconv::Layers::init()
{
  for (int k = 0; k < 7; ++k) {
    for (int c = 0; c < TSIZE; ++c) {
      int cMod = c % BOARDS;
      int locIdx = 0;
      int idx = 0;
      conv_global[k][c][idx] = -1;
      conv_rel_global[k][c][idx] = -1;
      for (int i = -k; i <= k; ++i) {
        for (int j = -k; j <= k; ++j) {
          int curr = c + i * BOARDS + j;
          if (curr >= 0 && curr < TSIZE) {
            if (j == 0) {
              conv_global[k][c][idx] = curr;
              conv_rel_global[k][c][idx] = locIdx;
              ++idx;
              conv_global[k][c][idx] = -1;
              conv_rel_global[k][c][idx] = -1;
            }
            if (j > 0 && curr % BOARDS > cMod) {
              conv_global[k][c][idx] = curr;
              conv_rel_global[k][c][idx] = locIdx;
              ++idx;
              conv_global[k][c][idx] = -1;
              conv_rel_global[k][c][idx] = -1;
            }
            if (j < 0 && curr % BOARDS < cMod) {
              conv_global[k][c][idx] = curr;
              conv_rel_global[k][c][idx] = locIdx;
              ++idx;
              conv_global[k][c][idx] = -1;
              conv_rel_global[k][c][idx] = -1;
            }
          }
          ++locIdx;
        }
      }
    }
  }
}

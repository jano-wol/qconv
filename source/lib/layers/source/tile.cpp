#include <layers/tile.h>

int qconv::layers::tileAbsolute[7][TSIZE][15 * 15 + 1];
int qconv::layers::tileRelative[7][TSIZE][15 * 15 + 1];

void qconv::layers::initTiles()
{
  for (int k = 0; k < 7; ++k) {
    for (int c = 0; c < TSIZE; ++c) {
      int cMod = c % BOARDS;
      int locIdx = 0;
      int idx = 0;
      tileAbsolute[k][c][idx] = -1;
      tileRelative[k][c][idx] = -1;
      for (int i = -k; i <= k; ++i) {
        for (int j = -k; j <= k; ++j) {
          int curr = c + i * BOARDS + j;
          if (curr >= 0 && curr < TSIZE) {
            if (j == 0) {
              tileAbsolute[k][c][idx] = curr;
              tileRelative[k][c][idx] = locIdx;
              ++idx;
              tileAbsolute[k][c][idx] = -1;
              tileRelative[k][c][idx] = -1;
            }
            if (j > 0 && curr % BOARDS > cMod) {
              tileAbsolute[k][c][idx] = curr;
              tileRelative[k][c][idx] = locIdx;
              ++idx;
              tileAbsolute[k][c][idx] = -1;
              tileRelative[k][c][idx] = -1;
            }
            if (j < 0 && curr % BOARDS < cMod) {
              tileAbsolute[k][c][idx] = curr;
              tileRelative[k][c][idx] = locIdx;
              ++idx;
              tileAbsolute[k][c][idx] = -1;
              tileRelative[k][c][idx] = -1;
            }
          }
          ++locIdx;
        }
      }
    }
  }
}

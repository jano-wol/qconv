#ifndef QCONV_LAYERS_TILE_H_INCLUDED
#define QCONV_LAYERS_TILE_H_INCLUDED

#include <layers/common.h>

constexpr int BOARDS = 20;
constexpr int TSIZE = (BOARDS * BOARDS);

namespace qconv::layers
{
extern int tileAbsolute[7][TSIZE][15 * 15 + 1];
extern int tileRelative[7][TSIZE][15 * 15 + 1];

void initTiles();
}  // namespace qconv::layers

#endif  // #ifndef QCONV_LAYERS_TILE_H_INCLUDED

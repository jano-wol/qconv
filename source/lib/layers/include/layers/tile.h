#ifndef QCONV_LAYERS_TILE_H_INCLUDED
#define QCONV_LAYERS_TILE_H_INCLUDED

#include <layers/common.h>

constexpr int BOARDS = 20;
constexpr int TSIZE = (BOARDS * BOARDS);

namespace qconv::Layers
{
extern int conv_global[7][TSIZE][15 * 15 + 1];
extern int conv_rel_global[7][TSIZE][15 * 15 + 1];

void init();
}  // namespace qconv::Layers

#endif  // #ifndef QCONV_LAYERS_TILE_H_INCLUDED

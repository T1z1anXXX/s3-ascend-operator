
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsinhTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);
  TILING_DATA_FIELD_DEF(uint64_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint64_t, block_size);
  TILING_DATA_FIELD_DEF(uint64_t, core_size);
  TILING_DATA_FIELD_DEF(uint64_t, core_remain);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Asinh, AsinhTilingData)
}

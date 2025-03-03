
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);

  TILING_DATA_FIELD_DEF(int32_t, y_dimensional);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_sumndarray);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Div, DivTilingData)
}

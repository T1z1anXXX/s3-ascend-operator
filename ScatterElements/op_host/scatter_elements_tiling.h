
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);

  TILING_DATA_FIELD_DEF(int32_t, dimensional);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, var_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, indices_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, updates_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, var_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, indices_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, updates_sumndarray);
  TILING_DATA_FIELD_DEF(int32_t, axis);
  TILING_DATA_FIELD_DEF(int32_t, reduce_id);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}

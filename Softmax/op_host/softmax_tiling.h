
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);

  TILING_DATA_FIELD_DEF(int32_t, dim);
  TILING_DATA_FIELD_DEF(int32_t, size);
  TILING_DATA_FIELD_DEF(int32_t, x_dimensional);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x_ndarray);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}


#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScatterElementsTilingData tiling;

  int32_t NUM = 11;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint32_t totalLength = context->GetInputTensor(0)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
        NUM = 11;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 9;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 10;
    }
    else{ //DT_FLOAT
        sizeofdatatype = 4;
        NUM = 9;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = totalLength - aivNum * core_size;

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    

    int32_t var_ndarray[20], indices_ndarray[20], updates_ndarray[20];
    int32_t dimensional;
    auto shape_var = context->GetInputShape(0)->GetOriginShape();
    auto shape_indices = context->GetInputShape(1)->GetOriginShape();
    auto shape_updates = context->GetInputShape(2)->GetOriginShape();

    dimensional =  shape_var.GetDimNum();

    for(int i = 0; i < dimensional; i++)
    {
        var_ndarray[i] = shape_var.GetDim(i);
        indices_ndarray[i] = shape_indices.GetDim(i);
        updates_ndarray[i] = shape_updates.GetDim(i);
    }
    
    tiling.set_dimensional(dimensional);
    tiling.set_var_ndarray(var_ndarray);
    tiling.set_indices_ndarray(indices_ndarray);
    tiling.set_updates_ndarray(updates_ndarray);

    int32_t var_sumndarray[20], indices_sumndarray[20], updates_sumndarray[20];
    var_sumndarray[dimensional-1] = 1;
    indices_sumndarray[dimensional-1] = 1;
    updates_sumndarray[dimensional-1] = 1;
    for(int i = dimensional-2; i >= 0; i--)
    {
        var_sumndarray[i] = var_sumndarray[i+1]*var_ndarray[i+1];
        indices_sumndarray[i] = indices_sumndarray[i+1]*indices_ndarray[i+1];
        updates_sumndarray[i] = updates_sumndarray[i+1]*updates_ndarray[i+1];
    }
    tiling.set_var_sumndarray(var_sumndarray);
    tiling.set_indices_sumndarray(indices_sumndarray);
    tiling.set_updates_sumndarray(updates_sumndarray);

    int32_t axis = *(context->GetAttrs()->GetInt(0));
    tiling.set_axis(axis);

    const char* reduce = context->GetAttrs()->GetStr(1);
    if(strcmp(reduce, "None") == 0){
        tiling.set_reduce_id(0);
    }
    else if (strcmp(reduce, "add") == 0){
        tiling.set_reduce_id(1);
    }
    else if(strcmp(reduce, "multiply") == 0){
        tiling.set_reduce_id(2);
    }
    else{
        tiling.set_reduce_id(0);
    }

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("None");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}

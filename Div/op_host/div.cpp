
#include "div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DivTilingData tiling;

    int32_t NUM = 9;
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
        NUM = 9;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 6;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 9;
    }
    else{ //DT_FLOAT
        sizeofdatatype = 4;
        NUM = 6;
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
    
    // broadcast
    uint32_t x1Size = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t x2Size = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t ySize = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();
    
    if(ySize != x1Size || ySize != x2Size)
    {
        context->SetTilingKey(2);

        int32_t y_ndarray[20], x1_ndarray[20], x2_ndarray[20];
        int32_t y_dimensional, x1_dimensional, x2_dimensional;
        auto shape_y = context->GetOutputShape(0)->GetOriginShape();
        auto shape_x1 = context->GetInputTensor(0)->GetOriginShape();
        auto shape_x2 = context->GetInputTensor(1)->GetOriginShape();

        y_dimensional =  shape_y.GetDimNum();
        x1_dimensional =  shape_x1.GetDimNum();
        x2_dimensional =  shape_x2.GetDimNum();

        int32_t max_y_dimensional;
        max_y_dimensional = y_dimensional;
        if(x1_dimensional > max_y_dimensional) max_y_dimensional = x1_dimensional;
        if(x2_dimensional > max_y_dimensional) max_y_dimensional = x2_dimensional;

        for(int i = 0; i < max_y_dimensional; i++)
        {
            if(i<y_dimensional){
                y_ndarray[y_dimensional-i-1] = shape_y.GetDim(i);
            } 
            else{
                y_ndarray[i] = 1;
            }                   
            if(i<x1_dimensional){
                x1_ndarray[x1_dimensional-i-1] = shape_x1.GetDim(i);
            } 
            else{
                x1_ndarray[i] = 1;
            }                    
            if(i<x2_dimensional){
                x2_ndarray[x2_dimensional-i-1] = shape_x2.GetDim(i);
            } 
            else{
                x2_ndarray[i] = 1;
            }                  
        }
        
        tiling.set_y_dimensional(max_y_dimensional);
        tiling.set_y_ndarray(y_ndarray);
        tiling.set_x1_ndarray(x1_ndarray);
        tiling.set_x2_ndarray(x2_ndarray);

        int32_t y_sumndarray[20], x1_sumndarray[20], x2_sumndarray[20];
        y_sumndarray[0] = 1;
        x1_sumndarray[0] = 1;
        x2_sumndarray[0] = 1;
        for(int i = 1; i <= max_y_dimensional; i++)
        {
            y_sumndarray[i] = y_sumndarray[i-1]*y_ndarray[i-1];
            x1_sumndarray[i] = x1_sumndarray[i-1]*x1_ndarray[i-1];
            x2_sumndarray[i] = x2_sumndarray[i-1]*x2_ndarray[i-1];
        }
        tiling.set_y_sumndarray(y_sumndarray);
        tiling.set_x1_sumndarray(x1_sumndarray);
        tiling.set_x2_sumndarray(x2_sumndarray);
    }
    else
    {
        context->SetTilingKey(1);
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
class Div : public OpDef {
public:
    explicit Div(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Div);
}

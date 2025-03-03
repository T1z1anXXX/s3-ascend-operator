#include "kernel_operator.h"
using namespace AscendC;
// constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_VAR, typename TYPE_INDICES, typename TYPE_UPDATES> class KernalScatterElements{
    using T = TYPE_VAR;
public:
    __aicore__ inline KernalScatterElements() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
                            int32_t dimensional, int32_t* var_ndarray, int32_t* indices_ndarray, int32_t* updates_ndarray,
                            int32_t* var_sumndarray, int32_t* indices_sumndarray, int32_t* updates_sumndarray, 
                            int32_t axis, int32_t reduce_id){

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        this->dimensional = dimensional;
        for(int k=0; k<=dimensional; k++)
        {
            this->var_ndarray[k] = var_ndarray[k];
            this->indices_ndarray[k] = indices_ndarray[k];
            this->updates_ndarray[k] = updates_ndarray[k];
            this->var_sumndarray[k] = var_sumndarray[k];
            this->indices_sumndarray[k] = indices_sumndarray[k];
            this->updates_sumndarray[k] = updates_sumndarray[k];
        }
        
        this->axis = axis;
        if(this->axis < 0){
            this->axis += dimensional;
        }
        this->reduce = reduce_id;
        printf( "reduction:    %d   \n", this->reduce);
        Gm_var.SetGlobalBuffer((__gm__ TYPE_VAR*)var + startPointer, var_sumndarray[0]*var_ndarray[0]);
        Gm_indices.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices + startPointer, indices_sumndarray[0]*indices_ndarray[0]);
        Gm_updates.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates + startPointer, updates_sumndarray[0]*updates_ndarray[0]);

        pipe.InitBuffer(B1, sizeof(float));
        pipe.InitBuffer(B2, sizeof(float));
        
    }

    __aicore__ inline void Process(){
        for(int i = 0; i < this->indices_sumndarray[0]*this->indices_ndarray[0]; i++){
            int32_t mdindex[20];
            int tmp = i;
            for(int idx = 0; idx < this->dimensional-1; idx++){
                mdindex[idx] = tmp / this->indices_sumndarray[idx];
                tmp %= this->indices_sumndarray[idx];
            }
            mdindex[this->dimensional-1] = tmp;
            // for(int idx = 0; idx < this->dimensional; idx++){
            //     printf(" %d ", mdindex[idx]);
            // }
            // printf("\n");
            int updatesIdx = 0;
            for(int idx = 0; idx < this->dimensional; idx++){
                updatesIdx += mdindex[idx]*this->updates_sumndarray[idx];
            }
            // printf(" updatesIdx: %d\n", updatesIdx);
            T updatesVal = (T)Gm_updates.GetValue(updatesIdx);
            TYPE_INDICES indicesVal = (TYPE_INDICES)Gm_indices.GetValue(i);
            
            // if(indicesVal<0){
            //     indicesVal += this->var_ndarray[this->axis];
            // }
            
            // printf(" indicesVal: %d\n", indicesVal);

            int varIdx = 0;
            for(int idx = 0; idx < this->dimensional; idx++){
                if(idx==this->axis){
                    varIdx += indicesVal*this->var_sumndarray[idx];
                }
                else{
                    varIdx += mdindex[idx]*this->var_sumndarray[idx];
                }
            }
            // printf(" varIdx: %d\n", varIdx);
            
            if(this->reduce==0){
                // printf("  none  \n");
                Gm_var.SetValue(varIdx, updatesVal);
            }
            else if(this->reduce==1){
                if constexpr (std::is_same_v<T, half>){
                    auto addVal = B1.Get<float>();
                    Duplicate(addVal, (float)Gm_var.GetValue(varIdx), 1);
                    Adds(addVal, addVal, (float)updatesVal, 1);
                    Gm_var.SetValue(varIdx, (T)addVal.GetValue(0));
                }
                else{
                    T addVal = updatesVal + (T)Gm_var.GetValue(varIdx);
                    Gm_var.SetValue(varIdx, addVal);
                }
            }
            else if(this->reduce==2){
                if constexpr (std::is_same_v<T, half>){
                    auto multiplyVal = B1.Get<float>();
                    Duplicate(multiplyVal, (float)Gm_var.GetValue(varIdx), 1);
                    Muls(multiplyVal, multiplyVal, (float)updatesVal, 1);
                    Gm_var.SetValue(varIdx, (T)multiplyVal.GetValue(0));
                }
                else{
                    T multiplyVal = updatesVal * (T)Gm_var.GetValue(varIdx);
                    Gm_var.SetValue(varIdx, multiplyVal);
                }
            }
            else{
                Gm_var.SetValue(varIdx, updatesVal);
            }
            
        }
    }

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> B1, B2;

    GlobalTensor<TYPE_VAR> Gm_var;
    GlobalTensor<TYPE_INDICES> Gm_indices;
    GlobalTensor<TYPE_UPDATES> Gm_updates;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    int32_t dimensional;
    int32_t var_ndarray[20];
    int32_t indices_ndarray[20];
    int32_t updates_ndarray[20];

    int32_t var_sumndarray[20];
    int32_t indices_sumndarray[20];
    int32_t updates_sumndarray[20];

    int32_t axis;
    int32_t reduce;
};

extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalScatterElements<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
    op.Init(var, indices, updates, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,
            tiling_data.dimensional, tiling_data.var_ndarray, tiling_data.indices_ndarray, tiling_data.updates_ndarray, 
            tiling_data.var_sumndarray, tiling_data.indices_sumndarray, tiling_data.updates_sumndarray, tiling_data.axis, tiling_data.reduce_id);
    op.Process();

    // TODO: user kernel impl
}
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelDiv{
    using T = TYPE_X1;
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(half));
        }

    }

    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length){
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length){
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            auto float_y = B_y.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Div(float_y, float_x1, float_x2, length);
            Cast(y, float_y, RoundMode::CAST_TRUNC, length);
        }
        else {
            if constexpr (std::is_same_v<T, int32_t>) {
                auto float_x1 = B_x1.Get<float>();
                auto float_x2 = B_x2.Get<float>();
                auto float_y = B_y.Get<float>();
                Cast(float_x1, x1, RoundMode::CAST_NONE, length);
                Cast(float_x2, x2, RoundMode::CAST_NONE, length);
                Div(float_y, float_x1, float_x2, length);
                Cast(y, float_y, RoundMode::CAST_TRUNC, length);
            }
            else{ //half float
                Div(y, x1, x2, length);
            }
        }

        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    TBuf<QuePosition::VECCALC> B_x1, B_x2, B_y;

    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

};

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelDivBroadcast{
    using T = TYPE_X1;
public:
    __aicore__ inline KernelDivBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, 
                            int32_t y_dimensional, int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray, 
                            int32_t *y_sumndarray, int32_t *x1_sumndarray, int32_t *x2_sumndarray){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, 1);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, 1);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        this->y_dimensional = y_dimensional;
        for(int k=0; k<=y_dimensional; k++)
        {
            this->y_ndarray[k] = y_ndarray[k];
            this->x1_ndarray[k] = x1_ndarray[k];
            this->x2_ndarray[k] = x2_ndarray[k];
            this->y_sumndarray[k] = y_sumndarray[k];
            this->x1_sumndarray[k] = x1_sumndarray[k];
            this->x2_sumndarray[k] = x2_sumndarray[k];
        }

        pipe.InitBuffer(Q_x1, BUFFER_NUM, 1 * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, 1 * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, 1 * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, 1 * sizeof(float));
            pipe.InitBuffer(B_x2, 1 * sizeof(float));
            pipe.InitBuffer(B_y, 1 * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, 1 * sizeof(half));
            pipe.InitBuffer(B_x2, 1 * sizeof(half));
            pipe.InitBuffer(B_y, 1 * sizeof(half));
        }
    }
    __aicore__ inline void Process(){
        LocalTensor<TYPE_X1> x1Local = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2Local = Q_x2.AllocTensor<TYPE_X2>();
        LocalTensor<TYPE_Y> yLocal = Q_y.AllocTensor<TYPE_Y>();

        int dim = this->y_dimensional;
        
        for(int j = 0; j < this->y_sumndarray[dim]; j++){
            int x1_start = 0, x2_start = 0;
            for(int k = 0; k < dim; k++){
                if(this->x1_ndarray[k] != 1){
                    x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
                if(this->x2_ndarray[k] != 1){
                    x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
            }
            TYPE_X1 x1 = Gm_x1.GetValue(x1_start);
            TYPE_X2 x2 = Gm_x2.GetValue(x2_start);
            x1Local.SetValue(0, (TYPE_X1)x1);
            x2Local.SetValue(0, (TYPE_X2)x2);
            
            
            if constexpr (std::is_same_v<T, int8_t>) {
                auto float_x1 = B_x1.Get<half>();
                auto float_x2 = B_x2.Get<half>();
                auto float_y = B_y.Get<half>();
                Cast(float_x1, x1Local, RoundMode::CAST_NONE, 1);
                Cast(float_x2, x2Local, RoundMode::CAST_NONE, 1);
                Div(float_y, float_x1, float_x2, 1);
                Cast(yLocal, float_y, RoundMode::CAST_TRUNC, 1);
            }
            else {
                if constexpr (std::is_same_v<T, int32_t>) {
                    auto float_x1 = B_x1.Get<float>();
                    auto float_x2 = B_x2.Get<float>();
                    auto float_y = B_y.Get<float>();
                    Cast(float_x1, x1Local, RoundMode::CAST_NONE, 1);
                    Cast(float_x2, x2Local, RoundMode::CAST_NONE, 1);
                    Div(float_y, float_x1, float_x2, 1);
                    Cast(yLocal, float_y, RoundMode::CAST_TRUNC, 1);
                }
                else{ //half float
                    Div(yLocal, x1Local, x2Local, 1);
                }
            }
            Gm_y.SetValue(j, (TYPE_Y)yLocal.GetValue(0));
        }
        Q_x1.FreeTensor(x1Local);
        Q_x2.FreeTensor(x2Local);
        Q_y.FreeTensor(yLocal);
    }
private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;

    TBuf<QuePosition::VECCALC> B_x1, B_x2, B_y;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    int32_t y_dimensional;
    int32_t y_ndarray[20];
    int32_t x1_ndarray[20];
    int32_t x2_ndarray[20];

    int32_t y_sumndarray[20];
    int32_t x1_sumndarray[20];
    int32_t x2_sumndarray[20];
};

extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if(TILING_KEY_IS(1)){
        KernelDiv<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){
        KernelDivBroadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, 
                tiling_data.y_dimensional, tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray, tiling_data.y_sumndarray, 
                tiling_data.x1_sumndarray, tiling_data.x2_sumndarray);
        op.Process();
    }
}
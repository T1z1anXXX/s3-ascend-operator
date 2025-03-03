#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class IsClose{
    using T = TYPE_X1;
public:
    __aicore__ inline IsClose() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, 
                            uint32_t core_remain, float rtol, float atol){
        
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

        this->rtol = rtol;
        this->atol = atol;

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
        pipe.InitBuffer(B_one, this->tileLength * sizeof(half));

        this->one = B_one.Get<half>();
        Duplicate(this->one, half(1), this->tileLength);

        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
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

        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        
        auto inty = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            
            Sub(float_x1, float_x1, float_x2, length);
            Abs(float_x1, float_x1, length);
            Abs(float_x2, float_x2, length);
            Muls(float_x2, float_x2, (half)this->rtol, length);
            Adds(float_x2, float_x2, (half)this->atol, length);

            Compare(bits, float_x1, float_x2, CMPMODE::LE, length);
            Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        }
        else {
            if constexpr (std::is_same_v<T, int32_t>) {
                auto float_x1 = B_x1.Get<float>();
                auto float_x2 = B_x2.Get<float>();
                Cast(float_x1, x1, RoundMode::CAST_NONE, length);
                Cast(float_x2, x2, RoundMode::CAST_NONE, length);
                
                Sub(float_x1, float_x1, float_x2, length);
                Abs(float_x1, float_x1, length);
                Abs(float_x2, float_x2, length);
                Muls(float_x2, float_x2, this->rtol, length);
                Adds(float_x2, float_x2, this->atol, length);

                Compare(bits, float_x1, float_x2, CMPMODE::LE, length);
                Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            }
            else if constexpr (std::is_same_v<T, float>){
                // |x1-x2|
                Sub(x1, x1, x2, length);
                Abs(x1, x1, length); 
                //atol+rtol*|x2|
                Abs(x2, x2, length);
                Muls(x2, x2, this->rtol, length);
                Adds(x2 ,x2, this->atol, length);

                Compare(bits, x1, x2, CMPMODE::LE, length);
                Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            }
            else{ //half
                // |x1-x2|
                Sub(x1, x1, x2, length);
                Abs(x1, x1, length); 
                //atol+rtol*|x2|
                Abs(x2, x2, length);
                Muls(x2, x2, (half)this->rtol, length);
                Adds(x2 ,x2, (half)this->atol, length);

                Compare(bits, x1, x2, CMPMODE::LE, length);
                Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            }
        }

        Cast(inty, result, RoundMode::CAST_ROUND, length);
        
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

    TBuf<QuePosition::VECCALC> B_result, B_one, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;

    LocalTensor<half> one;

    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    float rtol;
    float atol;
};

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class IsCloseBroadcast{
    using T = TYPE_X1;
public:
    __aicore__ inline IsCloseBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
                            int32_t y_dimensional, int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray, 
                            int32_t *y_sumndarray, int32_t *x1_sumndarray, int32_t *x2_sumndarray,
                            float rtol, float atol){
        
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

        this->rtol = rtol;
        this->atol = atol;

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

        pipe.InitBuffer(B_bits, 1 * sizeof(uint8_t));
        pipe.InitBuffer(B_result, 1 * sizeof(half));
        pipe.InitBuffer(B_one, 1 * sizeof(half));

        this->one = B_one.Get<half>();
        Duplicate(this->one, half(1), 1);

        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, 1 * sizeof(float));
            pipe.InitBuffer(B_x2, 1 * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, 1 * sizeof(half));
            pipe.InitBuffer(B_x2, 1 * sizeof(half));
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
            // printf("x1: %f ", x1);
            // printf("x2: %f \n", x2);
            x1Local.SetValue(0, (TYPE_X1)x1);
            x2Local.SetValue(0, (TYPE_X2)x2);
            
            auto bits = B_bits.Get<uint8_t>();
            auto result = B_result.Get<half>();
            
            auto inty = yLocal.template ReinterpretCast<uint8_t>();
            if constexpr (std::is_same_v<T, int8_t>) {
                auto float_x1 = B_x1.Get<half>();
                auto float_x2 = B_x2.Get<half>();
                Cast(float_x1, x1Local, RoundMode::CAST_NONE, 1);
                Cast(float_x2, x2Local, RoundMode::CAST_NONE, 1);
                
                Sub(float_x1, float_x1, float_x2, 1);
                Abs(float_x1, float_x1, 1);
                Abs(float_x2, float_x2, 1);
                Muls(float_x2, float_x2, (half)this->rtol, 1);
                Adds(float_x2, float_x2, (half)this->atol, 1);

                Compare(bits, float_x1, float_x2, CMPMODE::LE, 1);
                Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 1);
            }
            else {
                if constexpr (std::is_same_v<T, int32_t>) {
                    auto float_x1 = B_x1.Get<float>();
                    auto float_x2 = B_x2.Get<float>();
                    Cast(float_x1, x1Local, RoundMode::CAST_NONE, 1);
                    Cast(float_x2, x2Local, RoundMode::CAST_NONE, 1);
                    
                    Sub(float_x1, float_x1, float_x2, 1);
                    Abs(float_x1, float_x1, 1);
                    Abs(float_x2, float_x2, 1);
                    Muls(float_x2, float_x2, this->rtol, 1);
                    Adds(float_x2, float_x2, this->atol, 1);

                    Compare(bits, float_x1, float_x2, CMPMODE::LE, 1);
                    Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 1);
                }
                else if constexpr (std::is_same_v<T, float>){
                    // |x1-x2|
                    Sub(x1Local, x1Local, x2Local, 1);
                    Abs(x1Local, x1Local, 1); 
                    //atol+rtol*|x2|
                    Abs(x2Local, x2Local, 1);
                    Muls(x2Local, x2Local, this->rtol, 1);
                    Adds(x2Local ,x2Local, this->atol, 1);

                    Compare(bits, x1Local, x2Local, CMPMODE::LE, 1);
                    Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 1);
                }
                else{ //half
                    // |x1-x2|
                    Sub(x1Local, x1Local, x2Local, 1);
                    Abs(x1Local, x1Local, 1); 
                    //atol+rtol*|x2|
                    Abs(x2Local, x2Local, 1);
                    Muls(x2Local, x2Local, (half)this->rtol, 1);
                    Adds(x2Local ,x2Local, (half)this->atol, 1);

                    Compare(bits, x1Local, x2Local, CMPMODE::LE, 1);
                    Select(result, bits, one, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 1);
                }
            }

            Cast(inty, result, RoundMode::CAST_ROUND, 1);
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

    TBuf<QuePosition::VECCALC> B_result, B_one, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;

    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;

    LocalTensor<half> one;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    float rtol;
    float atol;

    int32_t y_dimensional;
    int32_t y_ndarray[20];
    int32_t x1_ndarray[20];
    int32_t x2_ndarray[20];

    int32_t y_sumndarray[20];
    int32_t x1_sumndarray[20];
    int32_t x2_sumndarray[20];
};

extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(1)){
        IsClose<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, 
                tiling_data.core_size, tiling_data.core_remain, tiling_data.rtol, tiling_data.atol);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){
        IsCloseBroadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, 
                tiling_data.y_dimensional, tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray, tiling_data.y_sumndarray, 
                tiling_data.x1_sumndarray, tiling_data.x2_sumndarray, tiling_data.rtol, tiling_data.atol);
        op.Process();
    }
}
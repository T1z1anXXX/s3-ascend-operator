#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y> class KernalSoftmax{
    using T = TYPE_X;
public:
    __aicore__ inline KernalSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
                            int32_t size, int32_t* x_ndarray, int32_t x_dimensional, int32_t dim){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        this->size = size;
        this->x_dimensional = x_dimensional;
        for(int i = 0; i < x_dimensional; i++){
            this->x_ndarray[i] = x_ndarray[i];
        }
        this->dim = dim;
        if(this->dim < 0){
            this->dim += this->x_dimensional;
        }

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, size);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, size);

        int32_t cycles = 1;
        int32_t interval = 1;
        int32_t loopCount = 1;

        for(int i = 0; i < this->dim; i++)
        {
            loopCount *= this->x_ndarray[i];
        }      

        cycles = this->x_ndarray[this->dim];

        for(int i = this->dim+1; i < this->x_dimensional; i++)
        {
            interval *= this->x_ndarray[i];
        }

        this->cycles = cycles;
        this->interval = interval;
        this->loopCount = loopCount;

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength*sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength*sizeof(TYPE_Y));
        pipe.InitBuffer(B_tmp, this->tileLength*sizeof(float));
    }

    __aicore__ inline void Process(){
        int32_t preloopCount = this->tileNum;
        for (int32_t i = 0; i < preloopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (preloopCount - 1);
        CopyIn(preloopCount - 1, length);
        Compute(preloopCount - 1, length);

        for(int z = 0; z < this->loopCount; z++)
        {   
            for(int j = 0; j < this->interval; j++){
                float expsum = 0;
                for(int i = 0; i < this->cycles; i++){
                    expsum += (float)Gm_x.GetValue(z*cycles*interval+i*interval+j);
                }
                for(int i = 0; i < this->cycles; i++){
                    Gm_y.SetValue(z*cycles*interval+i*interval+j, (float)Gm_x.GetValue(z*cycles*interval+i*interval+j)/expsum);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length){
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length){
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        LocalTensor<float> tmp = B_tmp.Get<float>();

        if constexpr (std::is_same_v<T, half>){
            Cast(tmp, x, RoundMode::CAST_NONE, length);
            Exp(tmp, tmp, length);
            Cast(y, tmp, RoundMode::CAST_NONE, length);
        }
        else{
            Exp(y, x, length);
        }
        Q_x.FreeTensor(x);
        Q_y.EnQue(y);
        y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_x[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){

    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_tmp;

    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    int32_t size;
    int32_t x_ndarray[20];
    int32_t x_dimensional;
    int32_t dim;

    int32_t cycles;
    int32_t interval;
    int32_t loopCount;
};


extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalSoftmax<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,
            tiling_data.size, tiling_data.x_ndarray, tiling_data.x_dimensional, tiling_data.dim);
    op.Process();
}
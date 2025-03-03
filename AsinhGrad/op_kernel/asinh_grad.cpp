#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z> class KernalAsinhGrad{
    using T = TYPE_Y;
public:
    __aicore__ inline KernalAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);
        Gm_dy.SetGlobalBuffer((__gm__ TYPE_DY*)dy + startPointer, bufferlength);
        Gm_z.SetGlobalBuffer((__gm__ TYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_dy, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_z, BUFFER_NUM, this->tileLength * sizeof(T));
        
        pipe.InitBuffer(B1, this->tileLength * sizeof(float));
        if constexpr(std::is_same_v<T, half>){
            pipe.InitBuffer(B_fy, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_fdy, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process(){
        if constexpr(std::is_same_v<T, float>){
            int32_t loopCount = this->tileNum;
            for(int32_t i = 0; i < loopCount - 1; i++){
                CopyIn(i, this->tileLength);
                Computefp32(i, this->tileLength);
                CopyOut(i, this->tileLength);
            }
            auto length = this->blockLength - this->tileLength*(loopCount-1);
            CopyIn(loopCount-1, length);
            Computefp32(loopCount-1, length);
            CopyOut(loopCount-1, length);
        }else if constexpr(std::is_same_v<T, half>){
            int32_t loopCount = this->tileNum;
            for(int32_t i = 0; i < loopCount - 1; i++){
                CopyIn(i, this->tileLength);
                Computefp16(i, this->tileLength);
                CopyOut(i, this->tileLength);
            }
            auto length = this->blockLength - this->tileLength*(loopCount-1);
            CopyIn(loopCount-1, length);
            Computefp16(loopCount-1, length);
            CopyOut(loopCount-1, length);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length){
        LocalTensor<T> yLocal = Q_y.AllocTensor<T>();
        LocalTensor<T> dyLocal = Q_dy.AllocTensor<T>();
        DataCopy(yLocal, Gm_y[progress * this->tileLength], length);
        DataCopy(dyLocal, Gm_dy[progress * this->tileLength], length);
        Q_y.EnQue(yLocal);
        Q_dy.EnQue(dyLocal);
    }

    __aicore__ inline void Computefp32(int32_t progress, uint32_t length){
        LocalTensor<T> yLocal = Q_y.DeQue<T>();
        LocalTensor<T> dyLocal = Q_dy.DeQue<T>();
        LocalTensor<T> zLocal = Q_z.AllocTensor<T>();
        LocalTensor<float> tmp = B1.Get<float>();
        Muls(tmp, yLocal, (float)(-1.0), length);
        Exp(yLocal, yLocal, length);
        Exp(tmp, tmp, length);
        Add(yLocal, yLocal, tmp, length);
        Muls(dyLocal, dyLocal, (float)(2.0), length);
        Div(zLocal, dyLocal, yLocal, length);

        Q_z.EnQue<T>(zLocal);
        Q_y.FreeTensor(yLocal);
        Q_dy.FreeTensor(dyLocal);
    }

    __aicore__ inline void Computefp16(int32_t progress, uint32_t length){
        LocalTensor<T> yLocal = Q_y.DeQue<T>();
        LocalTensor<T> dyLocal = Q_dy.DeQue<T>();
        LocalTensor<T> zLocal = Q_z.AllocTensor<T>();

        LocalTensor<float> fyLocal = B_fy.Get<float>();
        LocalTensor<float> fdyLocal = B_fdy.Get<float>();
        Cast(fyLocal, yLocal, RoundMode::CAST_NONE, length);
        Cast(fdyLocal, dyLocal, RoundMode::CAST_NONE, length);

        LocalTensor<float> tmp = B1.Get<float>();
        Muls(tmp, fyLocal, (float)(-1.0), length);
        Exp(fyLocal, fyLocal, length);
        Exp(tmp, tmp, length);
        Add(fyLocal, fyLocal, tmp, length);
        Muls(fdyLocal, fdyLocal, (float)(2.0), length);
        Div(fdyLocal, fdyLocal, fyLocal, length);

        Cast(zLocal, fdyLocal, RoundMode::CAST_NONE, length);

        Q_z.EnQue<T>(zLocal);
        Q_y.FreeTensor(yLocal);
        Q_dy.FreeTensor(dyLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){
        LocalTensor<T> zLocal = Q_z.DeQue<T>();
        DataCopy(Gm_z[progress * this->tileLength], zLocal, length);
        Q_z.FreeTensor(zLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_y, Q_dy;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_z;

    TBuf<QuePosition::VECCALC> B1;
    TBuf<QuePosition::VECCALC> B_fy, B_fdy;

    GlobalTensor<TYPE_Y> Gm_y;
    GlobalTensor<TYPE_DY> Gm_dy;
    GlobalTensor<TYPE_Z> Gm_z;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalAsinhGrad<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;
    op.Init(y, dy, z, tiling_data.totalLength, tiling_data.ALIGN_NUM,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
    // TODO: user kernel impl
}
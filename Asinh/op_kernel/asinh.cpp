#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y> class KernelAsinh{
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, 
                                uint64_t ALIGN_NUM, uint64_t block_size, 
                                uint64_t core_size, uint64_t core_remain){
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");

        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1? core_remain: 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM? ALIGN_NUM - this->blockLength%ALIGN_NUM: 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        // pipe.InitBuffer(tmpCal, this->tileLength * sizeof(float));

        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpbitBuffer1, (this->tileLength/16) * sizeof(uint16_t));
        pipe.InitBuffer(tmpbitBuffer2, (this->tileLength/16) * sizeof(uint16_t));

        pipe.InitBuffer(B_zero, this->tileLength * sizeof(float));
        this->zero = B_zero.Get<float>();
        Duplicate(this->zero, (float)(0.0), this->tileLength);

        pipe.InitBuffer(B_one, this->tileLength * sizeof(float));
        this->one = B_one.Get<float>();
        Duplicate(this->one, (float)(1.0), this->tileLength);

        if constexpr(!std::is_same_v<TYPE_X, float>){
            pipe.InitBuffer(B_fx, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_fy, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process(){
        if constexpr(std::is_same_v<TYPE_X, float>){
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
        }else if constexpr(std::is_same_v<TYPE_X, half>){
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
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Computefp16(int32_t progress, uint32_t length){
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        // LocalTensor<float> tmp = tmpCal.Get<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
        LocalTensor<uint16_t> tmpbit1 = tmpbitBuffer1.Get<uint16_t>();
        LocalTensor<uint16_t> tmpbit2 = tmpbitBuffer2.Get<uint16_t>();
        
        auto fxLocal = B_fx.Get<float>();
        auto fyLocal = B_fy.Get<float>();

        Cast(fxLocal, xLocal, RoundMode::CAST_NONE, length);
        
        
        // 获取大于0的元素bit掩码 tmpbit1:1代表大于0，0代表小于等于0
        Compare(tmpbit1, fxLocal, this->zero, CMPMODE::GT,length);
        // 获取小于0的元素bit掩码 tmpbit2:1代表小于0，0代表大于等于0
        Not(tmpbit2, tmpbit1, length/16);
        //Compare(tmpbit2,xLocal,this->zero,CMPMODE::LT,length);
        //获得大于0掩码
        Select(tmp1, tmpbit1, this->one, (float)(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        //获得小于0掩码
        Select(tmp2, tmpbit2, this->one, (float)(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        // Muls(tmp1,tmp1,(float)(-1.0),length);
        //获取sign（x）
        Sub(tmp1,tmp1,tmp2,length);



        // tmp = sqrt(x^2+1)
        Mul(fyLocal, fxLocal, fxLocal, length);
        Adds(fyLocal, fyLocal, static_cast<float>(1.0), length);
        Sqrt(fyLocal, fyLocal, length);

        //y = ln(x + tmp)
        Abs(fxLocal, fxLocal, length);
        Add(fyLocal, fxLocal, fyLocal, length);
        Ln(fyLocal, fyLocal, length); 
        Mul(fyLocal, fyLocal, tmp1, length);




        Cast(yLocal, fyLocal, RoundMode::CAST_NONE, length);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Computefp32(int32_t progress, uint32_t length){
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // LocalTensor<float> tmp = tmpCal.Get<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
        LocalTensor<uint16_t> tmpbit1 = tmpbitBuffer1.Get<uint16_t>();
        LocalTensor<uint16_t> tmpbit2 = tmpbitBuffer2.Get<uint16_t>();

        // 获取大于0的元素bit掩码 tmpbit1:1代表大于0，0代表小于等于0
        Compare(tmpbit1, xLocal, this->zero, CMPMODE::GT,length);
        // 获取小于0的元素bit掩码 tmpbit2:1代表小于0，0代表大于等于0
        Not(tmpbit2, tmpbit1, length/16);
        //Compare(tmpbit2,xLocal,this->zero,CMPMODE::LT,length);
        //获得大于0掩码
        Select(tmp1, tmpbit1, this->one, (float)(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        //获得小于0掩码
        Select(tmp2, tmpbit2, this->one, (float)(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        // Muls(tmp1,tmp1,(float)(-1.0),length);
        //获取sign（x）
        Sub(tmp1,tmp1,tmp2,length);


        // tmp = sqrt(x^2+1)
        Mul(yLocal, xLocal, xLocal, length);
        Adds(yLocal, yLocal, (float)(1.0), length);
        Sqrt(yLocal, yLocal, length);

        //y = ln(x + tmp)
        Abs(xLocal, xLocal, length);
        Add(yLocal, xLocal, yLocal, length);
        Ln(yLocal, yLocal, length);     
        Mul(yLocal, yLocal, tmp1, length); 

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    // TBuf<QuePosition::VECCALC> tmpCal;
    TBuf<QuePosition::VECCALC> B_fx, B_fy;
    TBuf<QuePosition::VECCALC> tmpbitBuffer1;
    TBuf<QuePosition::VECCALC> tmpbitBuffer2;
    TBuf<QuePosition::VECCALC> tmpBuffer1;
    TBuf<QuePosition::VECCALC> tmpBuffer2;

    TBuf<QuePosition::VECCALC> B_zero;
    TBuf<QuePosition::VECCALC> B_one;

    LocalTensor<float> zero;
    LocalTensor<float> one;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
};

extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinh<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
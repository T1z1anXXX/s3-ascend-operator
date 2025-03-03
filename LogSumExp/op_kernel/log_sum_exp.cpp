#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y> class KernalLogSumExp{
    using T = TYPE_X;
public:
    __aicore__ inline KernalLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->ALIGN_NUM = ALIGN_NUM;
        this->lastpadding = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startPointer, ALIGN_NUM);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(SumQueue, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(T));
        
    }

    __aicore__ inline void Process(){
        LocalTensor<T> sum = SumQueue.AllocTensor<T>();
        T zero = 0;
        Duplicate(sum, zero, this->tileLength);
        SumQueue.EnQue<T>(sum);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length - this->lastpadding);

        {
            LocalTensor<T> sum = SumQueue.DeQue<T>();
            LocalTensor<T> tmp = tmpBuffer.Get<T>();
       
            ReduceSum(sum, sum, tmp, this->tileLength);
            Ln(sum, sum, 1);

            SumQueue.EnQue<T>(sum);
            sum = SumQueue.DeQue<T>();
            DataCopy(yGm, sum, this->ALIGN_NUM);
            SumQueue.FreeTensor(sum);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        
        Exp(xLocal, xLocal, length);
        Add(sum, xLocal, sum, length);

        
        SumQueue.EnQue<T>(sum);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }


private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, 1> SumQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t ALIGN_NUM;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;

};

template<typename TYPE_X, typename TYPE_Y> class KernalLogSumExpDims{
    using T = TYPE_X;
public:
    __aicore__ inline KernalLogSumExpDims() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, 
                                uint32_t core_remain, uint32_t* reduce, uint32_t* shape, uint32_t dim){
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;

        this->reduce = reduce;
        this->shape = shape;
        this->dim = dim;
        this->ALIGN_NUM = ALIGN_NUM;
        this->ALIGN256 = ALIGN_NUM * 8;
        this->lastpadding = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        uint32_t outTotalLength = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                outTotalLength *= this->shape[i];
            }
        }
        outTotalLength =(outTotalLength + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startPointer, outTotalLength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(SumQueue, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ2, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(T));
        
    }

    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(0, i, this->tileLength);
            PreCal(i, this->tileLength);
            CopyPreOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(0, loopCount - 1, length);
        PreCal(loopCount - 1, length - this->lastpadding);
        CopyPreOut(loopCount - 1, length);

        uint32_t outTotalLength = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                outTotalLength *= this->shape[i];
            }
        }

        {
            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                InityGm(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            InityGm(loopCount - 1, length);
        }
        bool copyed = false;
        uint32_t sufDim = 1;
        uint32_t preDim = this->totalLength;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                preDim /= this->shape[i];
                sufDim *= this->shape[i];
            }else{
                if(sufDim == 1){
                    preDim /= this->shape[i];
                    auto length = (this->shape[i] + this->ALIGN256 - 1) / this->ALIGN256 * this->ALIGN256;
                    int32_t loopCount = (length + this->tileLength - 1) / this->tileLength;
                    for(int j=0;j<preDim;j++){
                        InitTensor(length < this->tileLength ? length : this->tileLength);
                        for(int k=0;k<loopCount-1;k++){
                            CopyIn(j * this->shape[i], k, this->tileLength);
                            Compute(i, this->tileLength);
                        }
                        auto L = this->shape[i] - this->tileLength * (loopCount - 1);

                        CopyIn(j * this->shape[i], loopCount - 1, (L + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM);
                        Compute(i, L);
                        CopyOut(j, length < this->tileLength ? length : this->tileLength);
                    }
                }else{
                    copyed = true;
                    auto length = (sufDim + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                    uint32_t loopCount = (length + this->tileLength - 1) / this->tileLength;
                    uint32_t d[11] = {0};
                    uint32_t dn[11] = {0};
                    d[i + 1] = dn[i + 1] = 1;
                    for(int k=i;k>=0;k--){
                        d[k] = d[k + 1] * this->shape[k];
                        if(this->reduce[k] == 0){
                            dn[k] = dn[k + 1] * this->shape[k];
                        }else{
                            dn[k] = dn[k + 1];
                        }
                    }
                    for(int j=0;j<preDim;j++){
                        uint32_t newp = 0;
                        for(int k=i;k>=0;k--){
                            if(this->reduce[k] == 0){
                                newp += dn[k + 1] * (j / d[k + 1] % this->shape[k]);
                            }
                        }

                        for(int k=0;k<loopCount-1;k++){
                            CopyIn2(j * sufDim, k, this->tileLength);
                            CopyValue(k, this->tileLength, 0);
                            CopyOut2(newp * sufDim, k, this->tileLength, 0);
                        }
                        auto L = sufDim - this->tileLength * (loopCount - 1);
                        auto L2 = (L + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                        CopyIn2(j * sufDim, loopCount - 1, L2);
                        CopyValue(loopCount - 1, L2, L2 - L);
                        CopyOut2(newp * sufDim, loopCount - 1, L2, L2 - L);
                    }

                    break;
                }
            }
        }
        if(!copyed){
            LocalTensor<T> sum = SumQueue.AllocTensor<T>();

            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyAnsIn(xGm, i, this->tileLength);
                CalAns(i, this->tileLength);
                CopyAns(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            CopyAnsIn(xGm, loopCount - 1, length);
            CalAns(loopCount - 1, length);
            CopyAns(loopCount - 1, length);

            SumQueue.FreeTensor(sum);
        }else{
            LocalTensor<T> sum = SumQueue.AllocTensor<T>();

            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyAnsIn(yGm, i, this->tileLength);
                CalAns(i, this->tileLength);
                CopyAns(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            CopyAnsIn(yGm, loopCount - 1, length);
            CalAns(loopCount - 1, length);
            CopyAns(loopCount - 1, length);

            SumQueue.FreeTensor(sum);
        }
    }
private:
    __aicore__ inline void InitTensor(uint32_t length)
    {
        LocalTensor<T> sum = SumQueue.AllocTensor<T>();
        T zero = 0;
        Duplicate(sum, zero, length);
        SumQueue.EnQue<T>(sum);
    }
    __aicore__ inline void InityGm(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        T zero = 0;
        Duplicate(zLocal, zero, length);
        outQueueZ.EnQue<T>(zLocal);
        zLocal = outQueueZ.DeQue<T>();
        DataCopy(yGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyIn(int32_t start, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[start + progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        
        
        Add(sum, xLocal, sum, length);
    
        
        SumQueue.EnQue<T>(sum);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t j, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ2.AllocTensor<T>();
        DataCopy(zLocal, xGm[j], this->ALIGN_NUM);
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        LocalTensor<T> tmp = tmpBuffer.Get<T>();
        
        ReduceSum(sum, sum, tmp, length);

        outQueueZ2.EnQue(zLocal);
        zLocal = outQueueZ2.DeQue<T>();
        
        zLocal.SetValue(0, sum.GetValue(0));
        
        outQueueZ2.EnQue(zLocal);
        zLocal = outQueueZ2.DeQue<T>();
        DataCopy(xGm[j], zLocal, this->ALIGN_NUM);
        //xGm.SetValue(j, sum.GetValue(0));
        SumQueue.FreeTensor(sum);
        outQueueZ2.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyIn2(int32_t start, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[start + progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyValue(int32_t progress, uint32_t length, uint32_t padding)
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        DataCopy(zLocal, xLocal, length);
        if(padding){
            T zero = 0;
            // Duplicate(xLocal, zero, padding);
            for(int i=length-padding;i<length;i++){
                zLocal.SetValue(i, zero);
            }
        }
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut2(int32_t start, int32_t progress, uint32_t length, uint32_t padding)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        
        SetAtomicAdd<T>();
        DataCopy(yGm[start + progress * this->tileLength], zLocal, length);
        SetAtomicNone();

        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyAnsIn(GlobalTensor<T> &Gm, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, Gm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CalAns(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        Ln(xLocal, xLocal, length);

        DataCopy(zLocal, xLocal, length);
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyAns(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(yGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void PreCal(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        
        
        Exp(zLocal, xLocal, length);

        
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyPreOut(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(xGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }


private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECOUT, 1> SumQueue, outQueueZ2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t totalLength;
    uint32_t ALIGN_NUM, ALIGN256;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;
    uint32_t* reduce;
    uint32_t* shape;
    uint32_t dim;

};

extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(1)){
        // printf("   all     ");
        KernalLogSumExp<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){
        // printf("   dims     ");
        KernalLogSumExpDims<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, 
                tiling_data.core_size, tiling_data.core_remain, tiling_data.reduce, tiling_data.shape, tiling_data.dim);
        op.Process();
    }

}
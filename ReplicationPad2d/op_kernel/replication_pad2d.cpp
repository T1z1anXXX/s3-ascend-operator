#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_PADDINGS, typename TYPE_Y> class KernalReplicationPad2d{
    using T = TYPE_X;
public:
    __aicore__ inline KernalReplicationPad2d() {}
    __aicore__ inline void Init(GM_ADDR paddings, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, 
                            int32_t lastdim, int32_t last2dim, int32_t num_last2dim){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->lastdim = lastdim;
        this->last2dim = last2dim;
        this->num_last2dim = num_last2dim;
        this->ALIGN_NUM = ALIGN_NUM;

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        
        Gm_paddings.SetGlobalBuffer((__gm__ TYPE_PADDINGS*)paddings + startPointer, 4);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
    }

    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, uint32_t core_size){
        for(int i = 0; i < 4; i++){
            TYPE_PADDINGS pad = Gm_paddings.GetValue(i);
            this->pad_num[i] = pad; 
        }
        int32_t lastdimnew = this->lastdim+this->pad_num[0]+this->pad_num[1];
        int32_t last2dimnew = this->last2dim+this->pad_num[2]+this->pad_num[3];


        auto startPointer = core_size * GetBlockIdx();
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, this->lastdim*this->last2dim*this->num_last2dim);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, lastdimnew*last2dimnew*this->num_last2dim);

        // pipe.InitBuffer(Q_x, BUFFER_NUM, 2 * length * sizeof(TYPE_X));
        // LocalTensor<TYPE_X> xLocal = Q_x.AllocTensor<TYPE_X>();
        
        for(int i = 0; i < this->num_last2dim; i++){
            for(int j = 0; j < this->last2dim; j++){
                TYPE_X left = 0;
                TYPE_X right = 0;
                for(int k = 0; k < this->lastdim; k++){
                    int idxsrc = i*this->last2dim*this->lastdim + j*this->lastdim + k;
                    int idxdst = i*last2dimnew*lastdimnew+(j+this->pad_num[2])*lastdimnew+(k+this->pad_num[0]);
                    TYPE_X val = (TYPE_X)Gm_x.GetValue(idxsrc);
                    if(k==0){
                        left = val;
                    }
                    else if(k == this->lastdim-1){
                        right = val;
                    }
                    Gm_y.SetValue(idxdst, val);
                }
                for(int k = 0; k < pad_num[0]; k++){
                    int idxdst = i*last2dimnew*lastdimnew+(j+this->pad_num[2])*lastdimnew+k;
                    Gm_y.SetValue(idxdst, left);
                }
                for(int k = 0; k < pad_num[1]; k++){
                    int idxdst = i*last2dimnew*lastdimnew+(j+this->pad_num[2])*lastdimnew+(k+this->lastdim+this->pad_num[0]);
                    Gm_y.SetValue(idxdst, right);
                }
            }
        }

        for(int i = 0; i < this->num_last2dim; i++){
            for(int j = 0; j < this->pad_num[2]; j++){
                for(int k = 0; k < this->pad_num[0]+this->lastdim+this->pad_num[1]; k++){
                    int idxsrc = i*last2dimnew*lastdimnew+this->pad_num[2]*lastdimnew+k;
                    int idxdst = i*last2dimnew*lastdimnew+j*lastdimnew+k;
                    Gm_y.SetValue(idxdst, (TYPE_X)Gm_y.GetValue(idxsrc));
                }
            }
            for(int j = 0; j < this->pad_num[3]; j++){
                for(int k = 0; k < this->pad_num[0]+this->lastdim+this->pad_num[1]; k++){
                    int idxsrc = i*last2dimnew*lastdimnew+(this->pad_num[2]+this->last2dim-1)*lastdimnew+k;
                    int idxdst = i*last2dimnew*lastdimnew+(j+this->pad_num[2]+this->last2dim)*lastdimnew+k;
                    Gm_y.SetValue(idxdst, (TYPE_X)Gm_y.GetValue(idxsrc));
                }
            }
        }

        // {
            // TYPE_X left = Gm_x.GetValue(0);
            // TYPE_X right = Gm_x.GetValue(this->lastdim-1);
            // DataCopy(xLocal[this->pad_num[0]], Gm_x[0], length);
            // DataCopy(xLocal[this->pad_num[0]], Gm_x[0], length);
            // DataCopy(Gm_y[0], xLocal, length);
            // for(int32_t j = 0; j < this->pad_num[0]; j++){
            //     xLocal.SetValue(j, (TYPE_X)left);
            // }
            // for(int32_t j = 0; j < this->pad_num[1]; j++){
            //     xLocal.SetValue(j + this->pad_num[0] + this->lastdim, (TYPE_X)right);
            // }
            // for(int32_t i = 0; i < this->pad_num[2]; i++){
            //     DataCopy(Gm_y[i * (this->lastdim + this->pad_num[0] + this->pad_num[1])], xLocal, length);
            // }
        // }

        // for (int32_t i = 0; i < this->num_lastdim; i++) {

        //     TYPE_X left = Gm_x.GetValue(i*this->lastdim);
        //     TYPE_X right = Gm_x.GetValue(i*this->lastdim + this->lastdim-1);
        //     DataCopy(xLocal[this->pad_num[0]], Gm_x[i*this->lastdim], length);
        //     for(int32_t j = 0; j < this->pad_num[0]; j++){
        //         xLocal.SetValue(j, (TYPE_X)left);
        //     }
        //     for(int32_t j = 0; j < this->pad_num[1]; j++){
        //         xLocal.SetValue(j + this->pad_num[0] + this->lastdim, (TYPE_X)right);
        //     }
        //     DataCopy(Gm_y[(i+this->pad_num[2]) * (this->lastdim + this->pad_num[0] + this->pad_num[1])], xLocal, length);
        // }

        // {
        //     TYPE_X left = Gm_x.GetValue(this->num_lastdim - 1);
        //     TYPE_X right = Gm_x.GetValue(this->num_lastdim - 1 + this->lastdim-1);
        //     DataCopy(xLocal[this->pad_num[0]], Gm_x[0], length);
        //     for(int32_t j = 0; j < this->pad_num[0]; j++){
        //         xLocal.SetValue(j, (TYPE_X)left);
        //     }
        //     for(int32_t j = 0; j < this->pad_num[1]; j++){
        //         xLocal.SetValue(j + this->pad_num[0] + this->lastdim, (TYPE_X)right);
        //     }
        //     for(int32_t i = 0; i < this->pad_num[3]; i++){
        //         DataCopy(Gm_y[(i+this->pad_num[2]+this->num_lastdim) * (this->lastdim + this->pad_num[0] + this->pad_num[1])], xLocal, length);
        //     }
        // }

        // Q_x.FreeTensor(xLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;

    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_PADDINGS> Gm_paddings;
    GlobalTensor<TYPE_Y> Gm_y;
    
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t ALIGN_NUM;
    
    int32_t lastdim;
    int32_t last2dim;
    int32_t num_last2dim;

    int32_t pad_num[4];
};

extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalReplicationPad2d<DTYPE_X, DTYPE_PADDINGS, DTYPE_Y> op;
    op.Init(paddings, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, 
            tiling_data.core_size, tiling_data.core_remain, tiling_data.lastdim, tiling_data.last2dim, tiling_data.num_last2dim);
    op.Process(x, y, tiling_data.core_size);
}
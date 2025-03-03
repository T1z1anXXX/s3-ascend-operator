#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernalNonMaxSuppression{
public:
    __aicore__ inline KernalNonMaxSuppression() {}
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold,
                            GM_ADDR score_threshold, GM_ADDR selected_indices, uint32_t totalLength, uint32_t ALIGN_NUM, 
                            uint32_t block_size, uint32_t core_size, uint32_t core_remain, int32_t num_batches, int32_t num_classes,
                            int32_t spatial_dimension, int32_t center_point_box){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        this->num_batches = num_batches;
        this->num_classes = num_classes;
        this->spatial_dimension = spatial_dimension;
        this->center_point_box = center_point_box;


        Gm_boxes.SetGlobalBuffer((__gm__ float*)boxes + startPointer, bufferlength);

        Gm_max_output_boxes_per_class.SetGlobalBuffer((__gm__ int32_t*)max_output_boxes_per_class + startPointer, 1);
        Gm_iou_threshold.SetGlobalBuffer((__gm__ float*)iou_threshold + startPointer, 1);
        Gm_score_threshold.SetGlobalBuffer((__gm__ float*)score_threshold + startPointer, 1);
        
        

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // this->ALING32_len = (this->spatial_dimension * sizeof(int32_t)+31)/32*32;
        pipe.InitBuffer(B_score_index, this->spatial_dimension * sizeof(int32_t));
        // pipe.InitBuffer(B_score, this->spatial_dimension * sizeof(float));
        // pipe.InitBuffer(B_sort, this->spatial_dimension * sizeof(float)*2);
        
    }

    __aicore__ inline void Process(GM_ADDR scores, GM_ADDR selected_indices, uint32_t core_size){
        auto startPointer = core_size * GetBlockIdx();
        this->output_per_class = Gm_max_output_boxes_per_class.GetValue(0);
        this->iou_thre = Gm_iou_threshold.GetValue(0);
        this->score_thre = Gm_score_threshold.GetValue(0);
        int32_t scores_shape = this->num_batches*this->num_classes*this->spatial_dimension;
        int32_t selected_shape = this->num_batches*this->num_classes*this->output_per_class*3;
        
        Gm_scores.SetGlobalBuffer((__gm__ float*)scores + startPointer, scores_shape);
        Gm_selected_indices.SetGlobalBuffer((__gm__ int32_t*)selected_indices + startPointer, selected_shape);
        
        int cnt = 0;
        for(uint32_t i = 0; i < this->num_batches; i++){
            for(uint32_t j = 0; j < this->num_classes; j++){
                
                int suppression[100];
                for(int32_t k = 0; k < this->spatial_dimension; k++){
                    uint32_t linearpos = k+j*this->spatial_dimension+i*this->spatial_dimension*this->num_classes;
                    float curscore = Gm_scores.GetValue(linearpos);
                    // printf("    cur score: %f", curscore);
                    if(curscore < this->score_thre){
                        suppression[k]=1;
                    }
                    else{
                        suppression[k]=0;
                    }
                    // printf("  suppression: %d  \n", suppression[k]);
                }
                for(uint32_t k = 0; k < this->output_per_class; k++){
                    float maxscore = -1;
                    uint32_t maxidx = -1;
                    for(uint32_t idx = 0; idx < this->spatial_dimension; idx++){
                        if(suppression[idx]){
                            continue;
                        }
                        uint32_t linearpos = idx+j*this->spatial_dimension+i*this->spatial_dimension*this->num_classes;
                        float curscore = Gm_scores.GetValue(linearpos);
                        if(curscore > maxscore){
                            maxscore = curscore;
                            maxidx = idx;
                        }
                    }

                    if(maxidx==-1){
                        continue;
                    }
                    // printf(" maxidx:%d  \n", maxidx);
                    Gm_selected_indices.SetValue(3*cnt, i);
                    Gm_selected_indices.SetValue(3*cnt+1, j);
                    Gm_selected_indices.SetValue(3*cnt+2, maxidx);
                    cnt++;

                    for(uint32_t idx = 0; idx < this->spatial_dimension; idx++){
                        if(suppression[idx]){
                            continue;
                        }

                        uint32_t linearpos1 = maxidx*4 + i*4*this->spatial_dimension;
                        float box1_y1 = Gm_boxes.GetValue(linearpos1);
                        float box1_x1 = Gm_boxes.GetValue(linearpos1+1);
                        float box1_y2 = Gm_boxes.GetValue(linearpos1+2);
                        float box1_x2 = Gm_boxes.GetValue(linearpos1+3);
                        
                        uint32_t linearpos2 = idx*4 + i*4*this->spatial_dimension;
                        float box2_y1 = Gm_boxes.GetValue(linearpos2);
                        float box2_x1 = Gm_boxes.GetValue(linearpos2+1);
                        float box2_y2 = Gm_boxes.GetValue(linearpos2+2);
                        float box2_x2 = Gm_boxes.GetValue(linearpos2+3);
                        
                        
                        float box1_y_min = box1_y1<box1_y2? box1_y1: box1_y2;
                        float box1_x_min = box1_x1<box1_x2? box1_x1: box1_x2;
                        float box1_y_max = box1_y1>box1_y2? box1_y1: box1_y2;
                        float box1_x_max = box1_x1>box1_x2? box1_x1: box1_x2;

                        float box2_y_min = box2_y1<box2_y2? box2_y1: box2_y2;
                        float box2_x_min = box2_x1<box2_x2? box2_x1: box2_x2;
                        float box2_y_max = box2_y1>box2_y2? box2_y1: box2_y2;
                        float box2_x_max = box2_x1>box2_x2? box2_x1: box2_x2;

                        float y_max = box1_y_min>box2_y_min? box1_y_min: box2_y_min;
                        float x_max = box1_x_min>box2_x_min? box1_x_min: box2_x_min;
                        float y_min = box1_y_max<box2_y_max? box1_y_max: box2_y_max;
                        float x_min = box1_x_max<box2_x_max? box1_x_max: box2_x_max;

                        float w = x_min - x_max;
                        float h = y_min - y_max;

                        float inter_area = 0;
                        if(w>0 && h>0){
                            inter_area = w*h;
                        }
                        
                        float area1 = (box1_x_max-box1_x_min)*(box1_y_max-box1_y_min);
                        float area2 = (box2_x_max-box2_x_min)*(box2_y_max-box2_y_min);
                        float union_area = area1 + area2 - inter_area;

                        float iou = inter_area/union_area;
                        // printf("   iou: %f\n ", iou);
                        if(iou > this->iou_thre){
                            suppression[idx] = 1;
                        }
                        
                    }
                }
                // for(uint32_t k = 0; k < (this->spatial_dimension+31)/32*32; k++){
                //     score_index.SetValue(k, k);
                // }

                // LocalTensor<float> score = B_score.Get<float>();
                // DataCopy(score, Gm_scores[(i*this->num_classes+j) * this->spatial_dimension], (this->spatial_dimension+31)/32*32);
                // for(int32_t k = this->spatial_dimension; k < (this->spatial_dimension+31)/32*32; k++){
                //     score.SetValue(k, (float)0);
                // }

                // // for(int32_t k = 0; k < (this->spatial_dimension+31)/32*32; k++){
                // //     printf("no.%d   value: %f   index: %d       ", k, score.GetValue(k), score_index.GetValue(k));
                // // }

                // LocalTensor<float> sort_res = B_sort.Get<float>();
                // Sort32<float>(sort_res, score, score_index, 0);

                // for(int k = 0; k < 2*(this->spatial_dimension+31)/32*32; k+=2){
                //     int index = static_cast<int>(sort_res.GetValue(k+1));
                //     printf(" index: %d", index);
                // }
                
            }
        }

        
    }
private:
    // __aicore__ inline void CopyIn(int32_t progress, uint32_t length){

    // }

    // __aicore__ inline void Compute(int32_t progress, uint32_t length){

    // }

    // __aicore__ inline void CopyOut(int32_t progress, uint32_t length){

    // }

private:
    TPipe pipe;

    // TQue<QuePosition::VECIN, BUFFER_NUM> Q_score;
    // TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    TBuf<QuePosition::VECCALC> B_score_index;
    // TBuf<QuePosition::VECCALC> B_score;
    // TBuf<QuePosition::VECCALC> B_sort;

    GlobalTensor<float> Gm_boxes;
    GlobalTensor<float> Gm_scores;
    GlobalTensor<int32_t> Gm_max_output_boxes_per_class;
    GlobalTensor<float> Gm_iou_threshold;
    GlobalTensor<float> Gm_score_threshold;
    GlobalTensor<int32_t> Gm_selected_indices;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    int32_t num_batches;
    int32_t num_classes;
    int32_t spatial_dimension;
    int32_t center_point_box;

    int32_t output_per_class;
    float iou_thre;
    float score_thre;

    // int32_t ALING32_len;
};

extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalNonMaxSuppression op;
    op.Init(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, selected_indices, tiling_data.totalLength, 
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.num_batches, 
            tiling_data.num_classes, tiling_data.spatial_dimension, tiling_data.center_point_box);
    op.Process(scores, selected_indices, tiling_data.core_size);
    // TODO: user kernel impl
}
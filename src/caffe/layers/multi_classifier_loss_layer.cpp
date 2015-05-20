#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    // TODO
}

INSTANTIATE_CLASS(MultiClassifierLossLayer);
REGISTER_LAYER_CLASS(MultiClassifierLoss);

}  // namespace caffe

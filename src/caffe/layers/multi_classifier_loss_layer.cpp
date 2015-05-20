#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiClassifierLossLayer::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    // TODO
}

template <typename Dtype>
void MultiClassifierLossLayer::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    // TODO
}

INSTANTIATE_CLASS(MultiClassifierLossLayer);
REGISTER_LAYER_CLASS(MultiClassifierLoss);

}  // namespace caffe

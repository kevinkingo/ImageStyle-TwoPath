#ifndef CAFFE_GRAM_LAYER_HPP_
#define CAFFE_GRAM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * 
 */
template <typename Dtype>
class GramLayer : public Layer<Dtype> {
 public:
  /**
   * 
   */
  explicit GramLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Gram"; }

 protected:
  /**
   * 
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * 
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //bool do_ave_;

  int down_channel_;
  int down_stride_;
  int old_num_slices_;
  int new_num_slices_;
  int slice_size_;

  int matrix_size_;
  int gram_size_;

  int batch_num_;

  Blob<Dtype> down_sampled_matrix_;
  Blob<Dtype> gram_matrix_;


};

}  // namespace caffe

#endif  // CAFFE_GRAM_LAYER_HPP_
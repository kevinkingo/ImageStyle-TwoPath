#include <vector>

#include "caffe/layers/gram_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const GramParameter& gram_param = this->layer_param_.gram_param();
	//do_ave_ = gram_param.do_ave();
	down_channel_ = gram_param.down_channel();

	const int bottom_channel = bottom[0]->shape(1);
	CHECK_EQ(bottom_channel % down_channel_, 0);
	down_stride_ = bottom_channel / down_channel_;

	old_num_slices_ = bottom[0]->count(0, 2);
	new_num_slices_ = old_num_slices_ / down_stride_;
	slice_size_ = bottom[0]->count(2);

	matrix_size_ = down_channel_ * slice_size_;
	gram_size_ = down_channel_ * down_channel_;

	batch_num_ = bottom[0]->shape(0);

	//LOG(INFO) << down_stride_ << " " << old_num_slices_ << " " << new_num_slices_ << " " << slice_size_ << " " << matrix_size_ << " " << gram_size_ << " " << batch_num_ << "\n"; 
}

template <typename Dtype>
void GramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape;
	top_shape.push_back(batch_num_);
	top_shape.push_back(down_channel_ * (down_channel_ + 1) / 2);
	top[0]->Reshape(top_shape);

	vector<int> down_sampled_shape = bottom[0]->shape();
	down_sampled_shape[1] = down_channel_;
	down_sampled_matrix_.Reshape(down_sampled_shape);

	vector<int> gram_shape = bottom[0]->shape();
	gram_shape[1] = down_channel_;
	gram_shape[2] = down_channel_;
	gram_matrix_.Reshape(gram_shape);

}

template <typename Dtype>
void GramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* down_sample_data = down_sampled_matrix_.mutable_cpu_data();
	
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, bottom_data + i * down_stride_ * slice_size_, down_sample_data + i * slice_size_);
	}
	/*
	//Averaging
	caffe_set<Dtype>(down_sampled_matrix_.count(), (Dtype)0, down_sample_data);
	for(int i = 0; i < old_num_slices_; i++) {
		caffe_axpy<Dtype>(slice_size_, (Dtype)(1.0 / down_stride_), bottom_data + i * slice_size_, down_sample_data + slice_size_ * (i / down_stride_));
	}
	*/

	Dtype* gram_data = gram_matrix_.mutable_cpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, down_channel_, down_channel_, slice_size_, (Dtype)1., down_sample_data + i * matrix_size_, down_sample_data + i * matrix_size_, (Dtype)0., gram_data + i * gram_size_);
	}

	Dtype* top_data = top[0]->mutable_cpu_data();
	int offset = 0;
	for(int i = 0; i < batch_num_; i++) {
		for(int j = 0; j < down_channel_; j++) {
			caffe_copy(j + 1, gram_data + i * gram_size_ + j * down_channel_, top_data + offset);
			offset += (j + 1);
		}
	}
	CHECK_EQ(offset, top[0]->count());

}

template <typename Dtype>
void GramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* gram_diff = gram_matrix_.mutable_cpu_diff();

	int offset = 0;
	caffe_set<Dtype>(gram_matrix_.count(), (Dtype)0, gram_diff);
	for(int i = 0; i < batch_num_; i++) {
		for(int j = 0; j < down_channel_; j++) {
			caffe_copy(j + 1, top_diff + offset, gram_diff + i * gram_size_ + j * down_channel_);
			for(int k = 0; k <= j; k++) {
				caffe_axpy<Dtype>(1, (Dtype)1, top_diff + offset + k, gram_diff + i * gram_size_ + k * down_channel_ + j);
			}
			offset += (j + 1);
		}
	}
	CHECK_EQ(offset, top[0]->count());
	
	Dtype* down_sample_diff = down_sampled_matrix_.mutable_cpu_diff();
	const Dtype* down_sample_data = down_sampled_matrix_.cpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, down_channel_, slice_size_, down_channel_, (Dtype)1., gram_diff + i * gram_size_, down_sample_data + i * matrix_size_, (Dtype)0., down_sample_diff + i * matrix_size_);
	}

	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set<Dtype>(bottom[0]->count(), 0, bottom_diff);
	
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, down_sample_diff + i * slice_size_, bottom_diff + i * down_stride_ * slice_size_);
	}
	/*
	//Averaging
	for(int i = 0; i < old_num_slices_; i++) {
		caffe_axpy<Dtype>(slice_size_, (Dtype)(1.0 / down_stride_), down_sample_diff + slice_size_ * (i / down_stride_), bottom_diff + i * slice_size_);
	}
	*/
}

#ifdef CPU_ONLY
STUB_GPU(GramLayer);
#endif

INSTANTIATE_CLASS(GramLayer);
REGISTER_LAYER_CLASS(Gram);

}  // namespace caffe
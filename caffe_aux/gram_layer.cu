#include <vector>

#include "caffe/layers/gram_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* down_sample_data = down_sampled_matrix_.mutable_gpu_data();
	
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, bottom_data + i * down_stride_ * slice_size_, down_sample_data + i * slice_size_);
	}
	/*
	//Averaging
	caffe_gpu_scal<Dtype>(down_sampled_matrix_.count(), (Dtype)0, down_sample_data);
	for(int i = 0; i < old_num_slices_; i++) {
		caffe_gpu_axpy<Dtype>(slice_size_, (Dtype)(1.0 / down_stride_), bottom_data + i * slice_size_, down_sample_data + slice_size_ * (i / down_stride_));
	}
	*/

	Dtype* gram_data = gram_matrix_.mutable_gpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, down_channel_, down_channel_, slice_size_, (Dtype)1., down_sample_data + i * matrix_size_, down_sample_data + i * matrix_size_, (Dtype)0., gram_data + i * gram_size_);
	}

	Dtype* top_data = top[0]->mutable_gpu_data();
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
void GramLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* gram_diff = gram_matrix_.mutable_gpu_diff();

	int offset = 0;
	caffe_gpu_scal<Dtype>(gram_matrix_.count(), (Dtype)0, gram_diff);
	for(int i = 0; i < batch_num_; i++) {
		for(int j = 0; j < down_channel_; j++) {
			caffe_copy(j + 1, top_diff + offset, gram_diff + i * gram_size_ + j * down_channel_);
			for(int k = 0; k <= j; k++) {
				caffe_gpu_axpy<Dtype>(1, (Dtype)1, top_diff + offset + k, gram_diff + i * gram_size_ + k * down_channel_ + j);
			}
			offset += (j + 1);
		}
	}
	CHECK_EQ(offset, top[0]->count());
	
	Dtype* down_sample_diff = down_sampled_matrix_.mutable_gpu_diff();
	const Dtype* down_sample_data = down_sampled_matrix_.gpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, down_channel_, slice_size_, down_channel_, (Dtype)1., gram_diff + i * gram_size_, down_sample_data + i * matrix_size_, (Dtype)0., down_sample_diff + i * matrix_size_);
	}

	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	caffe_gpu_scal<Dtype>(bottom[0]->count(), (Dtype)0, bottom_diff);
	
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, down_sample_diff + i * slice_size_, bottom_diff + i * down_stride_ * slice_size_);
	}
	/*
	//Averaging
	for(int i = 0; i < old_num_slices_; i++) {
		caffe_gpu_axpy<Dtype>(slice_size_, (Dtype)(1.0 / down_stride_), down_sample_diff + slice_size_ * (i / down_stride_), bottom_diff + i * slice_size_);
	}
	*/
}

INSTANTIATE_LAYER_GPU_FUNCS(GramLayer);


}  // namespace caffe
#ifndef CAFFE_COMPARISION_LAYER_HPP_
#define CAFFE_COMPARISION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Creates a "Comparison" path in the network by take 2 blobs
 *        as input and compare the difference between them then output the
 *        diff stats to file
 *
 * TODO(dox): add more paramter and functionality
 */
template <typename Dtype>
class ComparisonLayer: public Layer<Dtype> {
 public:
     friend class ConvolutionLayer<Dtype>;
     friend class SplitCConvolutionLayer<Dtype>;
     friend class WinogradConvolutionLayer<Dtype>;
     friend class BaseConvolutionLayer<Dtype>;
  explicit ComparisonLayer(const LayerParameter& param)
      : Layer<Dtype>(param), comp_param_(param.comp_param()) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Comparison"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  virtual inline bool NeedDumpTopBlobs() const { return false; }

  static void tell();
  inline void set_comp_param( const ComparisonParameter &param ) { comp_param_ = param; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  ComparisonParameter comp_param_;
  static vector<double> avg_diff_;
  static vector<double> max_diff_;
  static vector<double> min_diff_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

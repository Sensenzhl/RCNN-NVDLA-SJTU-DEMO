#ifndef CAFFE_VALIDATOR_LAYER_HPP_
#define CAFFE_VALIDATOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Creates a "Validator" path in the network to validate
 *  the bottom data based on configured rule
 *
 */
template <typename Dtype>
class ValidatorLayer: public Layer<Dtype> {
 public:
  explicit ValidatorLayer(const LayerParameter& param)
      : Layer<Dtype>(param), validator_param_(param.validator_param()) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Validator"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  virtual inline bool NeedDumpTopBlobs() const { return false; }
  inline void set_validator_param( const ValidatorParameter &param ) { validator_param_ = param; }

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
    ValidatorParameter validator_param_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

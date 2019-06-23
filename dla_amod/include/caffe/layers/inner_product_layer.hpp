#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/dump_layer.hpp"
#include "caffe/layers/loader_layer.hpp"
#include "caffe/layers/convertor_layer.hpp"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
#ifdef AMOD_CONFIG_ENA
      : Layer<Dtype>(param), weight_convert_(param),bias_convert_(param), output_truncat_(param),
      debug_loader_(param), debug_dump_(param) {
          if ( param.inner_product_param().has_weight_convert() )
              weight_convert_.set_convert_param(param.inner_product_param().weight_convert());
          if ( param.inner_product_param().has_bias_convert() )
              bias_convert_.set_convert_param(param.inner_product_param().bias_convert());
          if ( param.inner_product_param().has_output_truncat() )
              output_truncat_.set_convert_param(param.inner_product_param().output_truncat());
          if ( param.inner_product_param().has_debug_loader() )
              debug_loader_.set_loader_param(param.inner_product_param().debug_loader());
          if ( param.inner_product_param().has_debug_dump() )
              debug_dump_.set_dump_param(param.inner_product_param().debug_dump());

          weight_converted_ = false;
          bias_converted_ = false;
      }
#else
  : Layer<Dtype>(param) {}
#endif
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;
  void TruncatAccmulateInnerProduct(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void OptimizedInnerProduct(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

#ifdef AMOD_CONFIG_ENA
 private:
  ConvertorLayer<Dtype>               weight_convert_;
  ConvertorLayer<Dtype>               bias_convert_;
  ConvertorLayer<Dtype>               output_truncat_;
  bool                                weight_converted_;
  bool                                bias_converted_;
  LoaderLayer<Dtype>                  debug_loader_;
  DumpLayer<Dtype>                    debug_dump_;
#endif
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

#ifndef CAFFE_DUMP_LAYER_HPP_
#define CAFFE_DUMP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// FIXME: use friend is not a good idea
template <typename Dtype>
class ScaleLayer;
template <typename Dtype>
class BiasLayer;
template <typename Dtype>
class PReLULayer;
template <typename Dtype>
class ConvolutionLayer;
template <typename Dtype>
class SplitCConvolutionLayer;
template <typename Dtype>
class WinogradConvolutionLayer;
template <typename Dtype>
class BaseConvolutionLayer;
template <typename Dtype>
class BatchNormLayer;
template <typename Dtype>
class EltwiseLayer;
template <typename Dtype>
class DeconvolutionLayer;
template <typename Dtype>
class InnerProductLayer;

/**
 * @brief Creates a "dump" path in the network so that we can dump
 *      intermidate results to file
 *
 * TODO(dox): add more paramter and functionality
 */
template <typename Dtype>
class DumpLayer: public Layer<Dtype> {
 public:
     friend class ConvolutionLayer<Dtype>;
     friend class SplitCConvolutionLayer<Dtype>;
     friend class WinogradConvolutionLayer<Dtype>;
     friend class BaseConvolutionLayer<Dtype>;
     friend class BatchNormLayer<Dtype>;
     friend class ScaleLayer<Dtype>;
     friend class BiasLayer<Dtype>;
     friend class EltwiseLayer<Dtype>;
     friend class DeconvolutionLayer<Dtype>;
     friend class InnerProductLayer<Dtype>;
     friend class PReLULayer<Dtype>;
  explicit DumpLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
          dump_param = param.dump_param();
      }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "Dump"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 0; }
  virtual inline bool NeedDumpTopBlobs() const { return false; }
  
  inline void set_dump_param( const DumpParameter &param ) { 
      dump_param = param;
  }

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
  DumpParameter     dump_param;

  static uint32_t iter_id;
  static uint32_t batch_id;
  static uint32_t num_id;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

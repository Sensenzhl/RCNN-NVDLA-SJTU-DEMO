#ifndef CAFFE_CONVERTOR_LAYER_HPP_
#define CAFFE_CONVERTOR_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
#ifdef AMOD_CONFIG_ENA
template <typename Dtype>
class ConvolutionLayer;
template <typename Dtype>
class SplitCConvolutionLayer;
template <typename Dtype>
class WinogradConvolutionLayer;
template <typename Dtype>
class BaseConvolutionLayer;
template <typename Dtype>
class NVLRNLayer;
template <typename Dtype>
class NVSigmoidLayer;
template <typename Dtype>
class NVTanHLayer;
template <typename Dtype>
class InnerProductLayer;
template <typename Dtype>
class PoolingLayer;
template <typename Dtype>
class LUTLayer;
template <typename Dtype>
class SDPXLayer;
template <typename Dtype>
class EltwiseLayer;
template <typename Dtype>
class BatchNormLayer;
template <typename Dtype>
class SDPXLayer;
template <typename Dtype>
class DeconvolutionLayer;

/**
 * @brief Creates a "convertor" path in the network which take
 *      float32 as input then output the Customerized Data Type
 *      (CDT can be INT16/INT8/FP32)
 *
 * TODO(dox): add more paramter and functionality
 */
template <typename Dtype>
class ConvertorLayer: public Layer<Dtype> {
 public:
     friend class ConvolutionLayer<Dtype>;
     friend class SplitCConvolutionLayer<Dtype>;
     friend class WinogradConvolutionLayer<Dtype>;
     friend class BaseConvolutionLayer<Dtype>;
     friend class InnerProductLayer<Dtype>;
     friend class PoolingLayer<Dtype>;
     friend class LUTLayer<Dtype>;
     friend class NVLRNLayer<Dtype>;
     friend class NVSigmoidLayer<Dtype>;
     friend class NVTanHLayer<Dtype>;
     friend class EltwiseLayer<Dtype>;
     friend class BatchNormLayer<Dtype>;
     friend class SDPXLayer<Dtype>;
     friend class DeconvolutionLayer<Dtype>;
  explicit ConvertorLayer(const LayerParameter& param)
      : Layer<Dtype>(param), convert_param_(param.convert_param()) {
          ulOverflowCnt_ = 0;
          ulUnderflowCnt_ = 0;
          ulTotalCnt_ = 0;
      }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Convertor"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  inline void set_convert_param( const ConvertorParameter &param ) { convert_param_ = param; }
  inline const ConvertorParameter& get_convert_param( ) { return convert_param_; }

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
    void _convert_float2DBL(const Dtype *from, Dtype *to, uint32_t size, ConvertCoef &coef );
    void _convert_float2FP32(const Dtype *from, Dtype *to, uint32_t size, ConvertCoef &coef );
    void _convert_int2dbl(const Dtype *from, Dtype *to, uint32_t size, ConvertCoef &coef );
    void _copy(const Dtype *from, Dtype *to, uint32_t size );
    ConvertorParameter convert_param_;

    uint64_t    ulOverflowCnt_;
    uint64_t    ulUnderflowCnt_;
    uint64_t    ulTotalCnt_;
};
#endif
}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

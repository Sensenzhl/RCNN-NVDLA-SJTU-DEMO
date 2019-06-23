#ifndef CAFFE_WINOGRAD_CONV_LAYER_HPP_
#define CAFFE_WINOGRAD_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/convertor_layer.hpp"
#include "caffe/layers/dump_layer.hpp"
#include "caffe/layers/loader_layer.hpp"
#include "caffe/layers/comparision_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *       by winograd fashion
 */
template <typename Dtype>
class WinogradConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication), CUDNN (library
   *    kernels + stream parallelism) engines and split C convolution
   */
  explicit WinogradConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
            pra_weight_scale_ = 4;
            pra_feature_scale_ = 1;
      }

  virtual inline const char* type() const { return "WinogradConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:

#ifdef AMOD_CONFIG_ENA
    virtual void Weight_GetSum(Blob<Dtype>& sum_weight);
#endif
    void Winograd_Feature_PRA( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_Weight_PRA( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_Weight_INV_PRA( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_POA( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_Feature_PRA_Optimized( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_Weight_PRA_Optimized( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_Weight_INV_PRA_Optimized( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void Winograd_POA_Optimized( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top );
    void ChannelExtension( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, bool isWeight );

    void TruncatAccumulateConvolution(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    void OptimizedConvolution(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    vector<shared_ptr<Blob<Dtype> > >       remapped_kernels_;   // remapped kernels
    Dtype                                   pra_weight_scale_;
    Dtype                                   pra_feature_scale_;

    int                                     pra_feature_bits_;
    int                                     pra_weight_bits_;
    Blob<Dtype>                             fp16_max_exp_;
    static uint64_t                         ulWeightOverflowCnt_;
    static uint64_t                         ulWeightTotalCnt_;
    static uint64_t                         ulFeatureOverflowCnt_;
    static uint64_t                         ulFeatureTotalCnt_;
};
}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_

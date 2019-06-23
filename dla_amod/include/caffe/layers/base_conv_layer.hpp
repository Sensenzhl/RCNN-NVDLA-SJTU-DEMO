#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/convertor_layer.hpp"
#include "caffe/layers/dump_layer.hpp"
#include "caffe/layers/loader_layer.hpp"
#include "caffe/layers/comparision_layer.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
      output_truncat_(param ),
      weight_convert_(param), bias_convert_(param),
      debug_comp_(param) 
  {
          pad_val_ = 0;
          weight_converted_ = false;
          bias_converted_   = false;
          kernel_remapped_  = false;
          pra_done_         = false;

          if (param.convolution_param().has_pra_feature_truncat()) {
              pra_feature_truncat_.reset( new ConvertorLayer<Dtype>(param));
              pra_feature_truncat_->set_convert_param( param.convolution_param().pra_feature_truncat());
          }
          if (param.convolution_param().has_pra_weight_convert()) {
              pra_weight_convert_.reset( new ConvertorLayer<Dtype>(param));
              pra_weight_convert_->set_convert_param( param.convolution_param().pra_weight_convert());
          }
          output_truncat_.set_convert_param(param.convolution_param().output_truncat());
          if ( param.convolution_param().has_weight_convert() == true ) {
              weight_convert_.set_convert_param(param.convolution_param().weight_convert());
          }
          if ( param.convolution_param().has_bias_convert() == true ) {
              bias_convert_.set_convert_param(param.convolution_param().bias_convert());
          }
          if ( param.convolution_param().has_debug_comp() )
              debug_comp_.set_comp_param(param.convolution_param().debug_comp());
      }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  virtual bool is_need_dump(int x, int y, int c, ConvDebugDump dump_points);

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

#ifdef AMOD_CONFIG_ENA
  void load_data(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void dump_data(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void pre_process_converts(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void compare(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Weight_GetSum(Blob<Dtype>& sum_weight);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;      // S, R
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;            // stride_y, stride_x
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;               // pad_y, pad_x
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
                                    // CONV                 DECONV
  Blob<int> conv_input_shape_;      // C, in_h, in_w        K, out_h, out_w
  /// @brief The spatial dimensions of the col_buffer.
                                    // CONV                 DECONV
  vector<int> col_buffer_shape_;    // RSC, in_h, in_w      RSK, out_h, out_w
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;    // out_h, out_w
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;              // C*in_h*in_w
  int top_dim_;                 // K*out_h*out_w

  int channel_axis_;            // 1
  int num_;                     // N
  int channels_;                // C
  int group_;
  int out_spatial_dim_;         // W*H
  // CONV         DECONV
  int weight_offset_;           // K*C/g*S*R    K*C/g*S*R
  int num_output_;              // K
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

#ifdef AMOD_CONFIG_ENA
  bool need_c_extension_;
  Dtype pad_val_;
  bool weight_converted_;
  bool bias_converted_;
  bool kernel_remapped_;
  bool pra_done_;

  vector<shared_ptr<Blob<Dtype> > >       g_pra_wt_blobs_;
  vector<shared_ptr<Blob<Dtype> > >       g_pra_wt_last_blob_;

  ComparisonLayer<Dtype>                  debug_comp_;


  // convert the data after PRA to CDT to maximize MAC efficiency, however,
  // this will leads to precision loss, we nned to figure out how much loss
  // it will be;
  shared_ptr<ConvertorLayer<Dtype> >        pra_feature_truncat_;
  shared_ptr<ConvertorLayer<Dtype> >        pra_weight_convert_;
  ConvertorLayer<Dtype>                     weight_convert_;
  ConvertorLayer<Dtype>                     bias_convert_;

  // convert float32/double to CDT to save bandwidth(in A model, we still use DType
  // to store data to faciliate program, but the actual data range/precision are
  // limited to CDT to mimic hardware behavior)
  ConvertorLayer<Dtype>                   output_truncat_;

#endif
  Blob<Dtype> bias_multiplier_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff, Dtype pad_val) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff, pad_val);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif
                                    //  CONV                DECONV
  int num_kernels_im2col_;          // C*out_w*out_h        K*in_w*in_h                        
  int num_kernels_col2im_;          // C*in_h*in_w          K*out_h*out_w                        
  int conv_out_channels_;           // K                    C
  int conv_in_channels_;            // C                    K
  int conv_out_spatial_dim_;        // out_w*out_h          in_w*in_h
  int kernel_dim_;                  // C/g * S * R          K/g* S * R
  int col_offset_;                  // R*S*C*out_w*out_h/g  R*S*K*in_w*in_h/g
  int output_offset_;               // K*out_w*out_h/g      C*in_w*in_h/g

  Blob<Dtype> col_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_

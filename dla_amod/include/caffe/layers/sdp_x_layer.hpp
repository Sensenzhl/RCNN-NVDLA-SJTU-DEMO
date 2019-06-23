#ifndef CAFFE_SDPX_LAYER_HPP_
#define CAFFE_SDPX_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/convertor_layer.hpp"
#include "caffe/layers/lut_layer.hpp"

namespace caffe {
#ifdef AMOD_CONFIG_ENA

/**
 * @brief The SDP_X layer function, including:
 *
 * ALU: (MAX/MIN/SUM)
 *      Support per-cube, per-channel, per-element ops
 * MUL:
 *      Support per-cube, per-channel, per-element ops
 *
 * NOTE: does not implement Backwards operation.
 */
template <typename Dtype>
class SDPXLayer: public Layer<Dtype> {
 public:
  explicit SDPXLayer(const LayerParameter& param)
      : Layer<Dtype>(param), alu_cvt_(param), mul_cvt_(param), lut_(param){
          alu_index_ = -1;
          mul_index_ = -1;
          total_index_ = 0;
          param_cvt_    = false;
      }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SDPX"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  ConvertorLayer<Dtype> alu_cvt_;
  ConvertorLayer<Dtype> mul_cvt_;
  LUTLayer<Dtype> lut_;
  void ALUOp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void MULOp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void SaturateOutput(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  Dtype alu_val_;
  Dtype mul_val_;
  int   alu_index_;
  int   mul_index_;
  int   total_index_;
  bool  param_cvt_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

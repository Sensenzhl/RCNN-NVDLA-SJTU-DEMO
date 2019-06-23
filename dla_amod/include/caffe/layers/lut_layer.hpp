#ifndef CAFFE_LUT_LAYER_HPP_
#define CAFFE_LUT_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/convertor_layer.hpp"

namespace caffe {

template <typename Dtype>
class NVLRNLayer;
template <typename Dtype>
class NVSigmoidLayer;
template <typename Dtype>
class NVTanHLayer;
template <typename Dtype>
class SDPXLayer;

#ifdef AMOD_CONFIG_ENA
/**
 * @brief Creates a "LUT" path in the network which take
 *      float/double(actually, it's CDT) as input then output the
 *      corresponding value
 *
 */
template <typename Dtype>
class LUTLayer: public Layer<Dtype> {
 public:
     friend class NVLRNLayer<Dtype>;
     friend class NVSigmoidLayer<Dtype>;
     friend class NVTanHLayer<Dtype>;
     friend class SDPXLayer<Dtype>;
     virtual ~LUTLayer() {
         if ( x_table != NULL ) {
             delete [] x_table;
             x_table = NULL;
         }
         if ( y_table != NULL ) {
             delete [] y_table;
             y_table = NULL;
         }
     }
     explicit LUTLayer(const LayerParameter& param)
         : Layer<Dtype>(param),
         lut_convert_(param) {
        has_lut_param_   = false;
        has_func_        = false;
        func_            = NULL;
        is_initalized_   = false;

        uiXHitCnt_     = 0;
        uiYHitCnt_ = 0;
        uiOverflowCnt_   = 0;
        uiUnderflowCnt_  = 0;
        uiPriorityCnt_   = 0;
        uiTotalCnt_      = 0;

        x_table        = NULL;
        y_table    = NULL;

        is_fp16_         = false;

        num_frac_oflow_  = 0;
        num_frac_uflow_  = 0;
        is_sdp_          = false;
      }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual inline bool NeedDumpTopBlobs() const { return false; }

  typedef Dtype (*lut_func_t)(const Dtype in );

  virtual inline const char* type() const { return "LUT"; }
  inline void set_lut_param( const LUTParameter &param ) {
      lut_param_ = param;
      has_lut_param_ = true;

      lut_convert_.set_convert_param(param.lut_convert());
      if ( has_lut_param_ && has_func_ ) {
        _build_table();
      }
  }
  inline void set_func( lut_func_t func ) {
      func_ = func;
      has_func_ = true;

      if ( has_lut_param_ && has_func_ ) {
          _build_table();
      }
  }

  // TODO: I should decide whether or not put in convertor into lut

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
  void  _build_table(void);
  Dtype _lookup_y(const Dtype in, int8_t *status );
  Dtype _lookup_x(const Dtype in, int8_t *status );
  Dtype _lookup_y_int(const Dtype in, int8_t *status );
  Dtype _lookup_x_int(const Dtype in, int8_t *status );
  Dtype _lookup(const Dtype in );


  LUTParameter  lut_param_;
  bool          has_lut_param_;

  lut_func_t    func_;
  bool          has_func_;
  bool          is_initalized_;

  bool          is_fp16_;

  Dtype        *x_table;
  Dtype        *y_table;

  LUTParameter_PriSel  priority_;
  LUTParameter_PriSel  overflow_priority_;
  LUTParameter_PriSel  underflow_priority_;
  int           shifter_;

  LUTParameter_SymMethod sym_;
  Dtype         sym_x_;
  Dtype         sym_y_;

  LUTParameter_StepMethod x_method_;
  Dtype         receip_x_step_;
  Dtype         x_step_;
  double         x_start_index_;
  double         x_end_index_;

  Dtype         receip_y_step_;
  Dtype          y_step_;
  double         y_start_index_;
  double         y_end_index_;

  double         table_min_index_;
  double         table_max_index_;
  bool          has_overlap_;


  Dtype         max_diff_thresh_;

  Dtype         y_overflow_slope_;
  Dtype         y_overflow_shifter_;
  Dtype         y_underflow_slope_;
  Dtype         y_underflow_shifter_;
  Dtype         x_overflow_slope_;
  Dtype         x_overflow_shifter_;
  Dtype         x_underflow_slope_;
  Dtype         x_underflow_shifter_;

  ConvertorLayer<Dtype> lut_convert_;

  uint32_t      uiXHitCnt_;
  uint32_t      uiYHitCnt_;
  uint32_t      uiOverflowCnt_;
  uint32_t      uiUnderflowCnt_;
  uint32_t      uiPriorityCnt_;         // It can be treated as both hit or hybrid miss
  uint32_t      uiTotalCnt_;

  uint32_t      num_frac_oflow_;
  uint32_t      num_frac_uflow_;

  bool          is_sdp_;
};

#endif
}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_

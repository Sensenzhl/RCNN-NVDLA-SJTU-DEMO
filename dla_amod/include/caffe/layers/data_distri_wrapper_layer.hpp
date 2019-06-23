#ifndef CAFFE_DATADISTR_WRAPPER_CONV_LAYER_HPP_
#define CAFFE_DATADISTR_WRAPPER_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loader_layer.hpp"
#include "caffe/layers/nvlrn_layer.hpp"


namespace caffe {

/**
 * @brief Creates a "DataDistrWrapper" path in the network by load data blob from file
 *  then collect the data distribution
 *
 * TODO(dox): add more paramter and functionality
 */
template <typename Dtype>
class DataDistriWrapperLayer: public Layer<Dtype> {
 public:
  explicit DataDistriWrapperLayer(const LayerParameter& param)
      : Layer<Dtype>(param), datadistr_(param.datadistr_wrap_param().distr_param()), loader_(param), nv_lrn_(param)
  {
      datadistr_.set_layer_name( param.name() );
      datadistr_.set_datadist_param( param.datadistr_wrap_param().distr_param() );
      loader_.set_loader_param( param.datadistr_wrap_param().loader_param() );

      if (this->layer_param_.datadistr_wrap_param().has_nv_lrn_param() ) {
        nv_lrn_.set_lrn_param(this->layer_param_.datadistr_wrap_param().nv_lrn_param());
      }
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "DataDistriWrapper"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  virtual inline bool NeedDumpTopBlobs() const { return false; }

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
  DataDistributeLayer<Dtype>   datadistr_;
  LoaderLayer<Dtype>           loader_;
  NVLRNLayer<Dtype>             nv_lrn_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
